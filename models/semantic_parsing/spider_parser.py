import difflib
import os
from functools import partial
from typing import Dict, List, Tuple, Any, Mapping, Sequence

import sqlparse
import torch
from allennlp.common.util import pad_sequence_to_length
from allennlp.data import Vocabulary
from allennlp.data.fields.production_rule_field import ProductionRule, ProductionRuleArray
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, Seq2VecEncoder, Embedding, Attention, FeedForward, \
    TimeDistributed
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.nn import util, Activation
from allennlp.state_machines import BeamSearch
from allennlp.state_machines.states import GrammarStatelet
from torch_geometric.data import Data, Batch

from modules.gated_graph_conv import GatedGraphConv
from semparse.worlds.evaluate_spider import evaluate
from state_machines.states.rnn_statelet import RnnStatelet
from allennlp.state_machines.trainers import MaximumMarginalLikelihood
from allennlp.training.metrics import Average
from overrides import overrides

from semparse.contexts.spider_context_utils import action_sequence_to_sql
from semparse.worlds.spider_world import SpiderWorld
from state_machines.states.grammar_based_state import GrammarBasedState
from state_machines.states.sql_state import SqlState
from state_machines.transition_functions.attend_past_schema_items_transition import \
    AttendPastSchemaItemsTransitionFunction
from state_machines.transition_functions.linking_transition_function import LinkingTransitionFunction


@Model.register("spider")
class SpiderParser(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 encoder: Seq2SeqEncoder,
                 entity_encoder: Seq2VecEncoder,
                 decoder_beam_search: BeamSearch,
                 question_embedder: TextFieldEmbedder,
                 input_attention: Attention,
                 past_attention: Attention,
                 max_decoding_steps: int,
                 action_embedding_dim: int,
                 gnn: bool = True,
                 decoder_use_graph_entities: bool = True,
                 decoder_self_attend: bool = True,
                 gnn_timesteps: int = 2,
                 parse_sql_on_decoding: bool = True,
                 add_action_bias: bool = True,
                 use_neighbor_similarity_for_linking: bool = True,
                 dataset_path: str = 'dataset',
                 training_beam_size: int = None,
                 decoder_num_layers: int = 1,
                 dropout: float = 0.0,
                 rule_namespace: str = 'rule_labels',
                 scoring_dev_params: dict = None,
                 debug_parsing: bool = False) -> None:
        super().__init__(vocab)
        self.vocab = vocab
        self._encoder = encoder
        self._max_decoding_steps = max_decoding_steps
        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x
        self._rule_namespace = rule_namespace
        self._question_embedder = question_embedder
        self._add_action_bias = add_action_bias
        self._scoring_dev_params = scoring_dev_params or {}
        self.parse_sql_on_decoding = parse_sql_on_decoding
        self._entity_encoder = TimeDistributed(entity_encoder)
        self._use_neighbor_similarity_for_linking = use_neighbor_similarity_for_linking
        self._self_attend = decoder_self_attend
        self._decoder_use_graph_entities = decoder_use_graph_entities

        self._action_padding_index = -1  # the padding value used by IndexField

        self._exact_match = Average()
        self._sql_evaluator_match = Average()
        self._action_similarity = Average()
        self._acc_single = Average()
        self._acc_multi = Average()
        self._beam_hit = Average()

        self._action_embedding_dim = action_embedding_dim

        num_actions = vocab.get_vocab_size(self._rule_namespace)
        if self._add_action_bias:
            input_action_dim = action_embedding_dim + 1
        else:
            input_action_dim = action_embedding_dim
        self._action_embedder = Embedding(num_embeddings=num_actions, embedding_dim=input_action_dim)
        self._output_action_embedder = Embedding(num_embeddings=num_actions, embedding_dim=action_embedding_dim)

        encoder_output_dim = encoder.get_output_dim()
        if gnn:
            encoder_output_dim += action_embedding_dim

        self._first_action_embedding = torch.nn.Parameter(torch.FloatTensor(action_embedding_dim))
        self._first_attended_utterance = torch.nn.Parameter(torch.FloatTensor(encoder_output_dim))
        self._first_attended_output = torch.nn.Parameter(torch.FloatTensor(action_embedding_dim))
        torch.nn.init.normal_(self._first_action_embedding)
        torch.nn.init.normal_(self._first_attended_utterance)
        torch.nn.init.normal_(self._first_attended_output)

        self._num_entity_types = 9
        self._embedding_dim = question_embedder.get_output_dim()

        self._entity_type_encoder_embedding = Embedding(self._num_entity_types, self._embedding_dim)
        self._entity_type_decoder_embedding = Embedding(self._num_entity_types, action_embedding_dim)

        self._linking_params = torch.nn.Linear(16, 1)
        torch.nn.init.uniform_(self._linking_params.weight, 0, 1)

        num_edge_types = 3
        self._gnn = GatedGraphConv(self._embedding_dim, gnn_timesteps, num_edge_types=num_edge_types, dropout=dropout)

        self._decoder_num_layers = decoder_num_layers

        self._beam_search = decoder_beam_search
        self._decoder_trainer = MaximumMarginalLikelihood(training_beam_size)

        if decoder_self_attend:
            self._transition_function = AttendPastSchemaItemsTransitionFunction(encoder_output_dim=encoder_output_dim,
                                                                                action_embedding_dim=action_embedding_dim,
                                                                                input_attention=input_attention,
                                                                                past_attention=past_attention,
                                                                                predict_start_type_separately=False,
                                                                                add_action_bias=self._add_action_bias,
                                                                                dropout=dropout,
                                                                                num_layers=self._decoder_num_layers)
        else:
            self._transition_function = LinkingTransitionFunction(encoder_output_dim=encoder_output_dim,
                                                                  action_embedding_dim=action_embedding_dim,
                                                                  input_attention=input_attention,
                                                                  predict_start_type_separately=False,
                                                                  add_action_bias=self._add_action_bias,
                                                                  dropout=dropout,
                                                                  num_layers=self._decoder_num_layers)

        self._ent2ent_ff = FeedForward(action_embedding_dim, 1, action_embedding_dim, Activation.by_name('relu')())

        self._neighbor_params = torch.nn.Linear(self._embedding_dim, self._embedding_dim)

        # TODO: Remove hard-coded dirs
        self._evaluate_func = partial(evaluate,
                                      db_dir=os.path.join(dataset_path, 'database'),
                                      table=os.path.join(dataset_path, 'tables.json'),
                                      check_valid=False)

        self.debug_parsing = debug_parsing

    @overrides
    def forward(self,  # type: ignore
                utterance: Dict[str, torch.LongTensor],
                valid_actions: List[List[ProductionRule]],
                world: List[SpiderWorld],
                schema: Dict[str, torch.LongTensor],
                action_sequence: torch.LongTensor = None) -> Dict[str, torch.Tensor]:

        batch_size = len(world)
        device = utterance['tokens'].device

        initial_state = self._get_initial_state(utterance, world, schema, valid_actions)

        if action_sequence is not None:
            # Remove the trailing dimension (from ListField[ListField[IndexField]]).
            action_sequence = action_sequence.squeeze(-1)
            action_mask = action_sequence != self._action_padding_index
        else:
            action_mask = None

        if self.training:
            decode_output = self._decoder_trainer.decode(initial_state,
                                                         self._transition_function,
                                                         (action_sequence.unsqueeze(1), action_mask.unsqueeze(1)))

            return {'loss': decode_output['loss']}
        else:
            loss = torch.tensor([0]).float().to(device)
            if action_sequence is not None and action_sequence.size(1) > 1:
                try:
                    loss = self._decoder_trainer.decode(initial_state,
                                                        self._transition_function,
                                                        (action_sequence.unsqueeze(1),
                                                         action_mask.unsqueeze(1)))['loss']
                except ZeroDivisionError:
                    # reached a dead-end during beam search
                    pass

            outputs: Dict[str, Any] = {
                'loss': loss
            }

            num_steps = self._max_decoding_steps
            # This tells the state to start keeping track of debug info, which we'll pass along in
            # our output dictionary.
            initial_state.debug_info = [[] for _ in range(batch_size)]

            best_final_states = self._beam_search.search(num_steps,
                                                         initial_state,
                                                         self._transition_function,
                                                         keep_final_unfinished_states=False)

            self._compute_validation_outputs(valid_actions,
                                             best_final_states,
                                             world,
                                             action_sequence,
                                             outputs)
            return outputs

    def _get_initial_state(self,
                           utterance: Dict[str, torch.LongTensor],
                           worlds: List[SpiderWorld],
                           schema: Dict[str, torch.LongTensor],
                           actions: List[List[ProductionRule]]) -> GrammarBasedState:
        schema_text = schema['text']
        embedded_schema = self._question_embedder(schema_text, num_wrapping_dims=1)
        schema_mask = util.get_text_field_mask(schema_text, num_wrapping_dims=1).float()

        embedded_utterance = self._question_embedder(utterance)
        utterance_mask = util.get_text_field_mask(utterance).float()

        batch_size, num_entities, num_entity_tokens, _ = embedded_schema.size()
        num_entities = max([len(world.db_context.knowledge_graph.entities) for world in worlds])
        num_question_tokens = utterance['tokens'].size(1)

        # entity_types: tensor with shape (batch_size, num_entities), where each entry is the
        # entity's type id.
        # entity_type_dict: Dict[int, int], mapping flattened_entity_index -> type_index
        # These encode the same information, but for efficiency reasons later it's nice
        # to have one version as a tensor and one that's accessible on the cpu.
        entity_types, entity_type_dict = self._get_type_vector(worlds, num_entities, embedded_schema.device)

        entity_type_embeddings = self._entity_type_encoder_embedding(entity_types)

        # Compute entity and question word similarity.  We tried using cosine distance here, but
        # because this similarity is the main mechanism that the model can use to push apart logit
        # scores for certain actions (like "n -> 1" and "n -> -1"), this needs to have a larger
        # output range than [-1, 1].
        question_entity_similarity = torch.bmm(embedded_schema.view(batch_size,
                                                                    num_entities * num_entity_tokens,
                                                                    self._embedding_dim),
                                               torch.transpose(embedded_utterance, 1, 2))

        question_entity_similarity = question_entity_similarity.view(batch_size,
                                                                     num_entities,
                                                                     num_entity_tokens,
                                                                     num_question_tokens)
        # (batch_size, num_entities, num_question_tokens)
        question_entity_similarity_max_score, _ = torch.max(question_entity_similarity, 2)

        # (batch_size, num_entities, num_question_tokens, num_features)
        linking_features = schema['linking']

        linking_scores = question_entity_similarity_max_score

        feature_scores = self._linking_params(linking_features).squeeze(3)

        linking_scores = linking_scores + feature_scores

        # (batch_size, num_question_tokens, num_entities)
        linking_probabilities = self._get_linking_probabilities(worlds, linking_scores.transpose(1, 2),
                                                                utterance_mask, entity_type_dict)

        # (batch_size, num_entities, num_neighbors) or None
        neighbor_indices = self._get_neighbor_indices(worlds, num_entities, linking_scores.device)

        if self._use_neighbor_similarity_for_linking and neighbor_indices is not None:
            # (batch_size, num_entities, embedding_dim)
            encoded_table = self._entity_encoder(embedded_schema, schema_mask)

            # Neighbor_indices is padded with -1 since 0 is a potential neighbor index.
            # Thus, the absolute value needs to be taken in the index_select, and 1 needs to
            # be added for the mask since that method expects 0 for padding.
            # (batch_size, num_entities, num_neighbors, embedding_dim)
            embedded_neighbors = util.batched_index_select(encoded_table, torch.abs(neighbor_indices))

            neighbor_mask = util.get_text_field_mask({'ignored': neighbor_indices + 1},
                                                     num_wrapping_dims=1).float()

            # Encoder initialized to easily obtain a masked average.
            neighbor_encoder = TimeDistributed(BagOfEmbeddingsEncoder(self._embedding_dim, averaged=True))
            # (batch_size, num_entities, embedding_dim)
            embedded_neighbors = neighbor_encoder(embedded_neighbors, neighbor_mask)
            projected_neighbor_embeddings = self._neighbor_params(embedded_neighbors.float())

            # (batch_size, num_entities, embedding_dim)
            entity_embeddings = torch.tanh(entity_type_embeddings + projected_neighbor_embeddings)
        else:
            # (batch_size, num_entities, embedding_dim)
            entity_embeddings = torch.tanh(entity_type_embeddings)

        link_embedding = util.weighted_sum(entity_embeddings, linking_probabilities)
        encoder_input = torch.cat([link_embedding, embedded_utterance], 2)

        # (batch_size, utterance_length, encoder_output_dim)
        encoder_outputs = self._dropout(self._encoder(encoder_input, utterance_mask))

        max_entities_relevance = linking_probabilities.max(dim=1)[0]
        entities_relevance = max_entities_relevance.unsqueeze(-1).detach()

        graph_initial_embedding = entity_type_embeddings * entities_relevance

        encoder_output_dim = self._encoder.get_output_dim()
        if self._gnn:
            entities_graph_encoding = self._get_schema_graph_encoding(worlds,
                                                                      graph_initial_embedding)
            graph_link_embedding = util.weighted_sum(entities_graph_encoding, linking_probabilities)
            encoder_outputs = torch.cat((
                encoder_outputs,
                graph_link_embedding
            ), dim=-1)
            encoder_output_dim = self._action_embedding_dim + self._encoder.get_output_dim()
        else:
            entities_graph_encoding = None

        if self._self_attend:
            # linked_actions_linking_scores = self._get_linked_actions_linking_scores(actions, entities_graph_encoding)
            entities_ff = self._ent2ent_ff(entities_graph_encoding)
            linked_actions_linking_scores = torch.bmm(entities_ff, entities_ff.transpose(1, 2))
        else:
            linked_actions_linking_scores = [None] * batch_size

        # This will be our initial hidden state and memory cell for the decoder LSTM.
        final_encoder_output = util.get_final_encoder_states(encoder_outputs,
                                                             utterance_mask,
                                                             self._encoder.is_bidirectional())
        memory_cell = encoder_outputs.new_zeros(batch_size, encoder_output_dim)
        initial_score = embedded_utterance.data.new_zeros(batch_size)

        # To make grouping states together in the decoder easier, we convert the batch dimension in
        # all of our tensors into an outer list.  For instance, the encoder outputs have shape
        # `(batch_size, utterance_length, encoder_output_dim)`.  We need to convert this into a list
        # of `batch_size` tensors, each of shape `(utterance_length, encoder_output_dim)`.  Then we
        # won't have to do any index selects, or anything, we'll just do some `torch.cat()`s.
        initial_score_list = [initial_score[i] for i in range(batch_size)]
        encoder_output_list = [encoder_outputs[i] for i in range(batch_size)]
        utterance_mask_list = [utterance_mask[i] for i in range(batch_size)]
        initial_rnn_state = []
        for i in range(batch_size):
            initial_rnn_state.append(RnnStatelet(final_encoder_output[i],
                                                 memory_cell[i],
                                                 self._first_action_embedding,
                                                 self._first_attended_utterance,
                                                 encoder_output_list,
                                                 utterance_mask_list))

        initial_grammar_state = [self._create_grammar_state(worlds[i],
                                                            actions[i],
                                                            linking_scores[i],
                                                            linked_actions_linking_scores[i],
                                                            entity_types[i],
                                                            entities_graph_encoding[
                                                                i] if entities_graph_encoding is not None else None)
                                 for i in range(batch_size)]

        initial_sql_state = [SqlState(actions[i], self.parse_sql_on_decoding) for i in range(batch_size)]

        initial_state = GrammarBasedState(batch_indices=list(range(batch_size)),
                                          action_history=[[] for _ in range(batch_size)],
                                          score=initial_score_list,
                                          rnn_state=initial_rnn_state,
                                          grammar_state=initial_grammar_state,
                                          sql_state=initial_sql_state,
                                          possible_actions=actions,
                                          action_entity_mapping=[w.get_action_entity_mapping() for w in worlds])

        return initial_state

    @staticmethod
    def _get_neighbor_indices(worlds: List[SpiderWorld],
                              num_entities: int,
                              device: torch.device) -> torch.LongTensor:
        """
        This method returns the indices of each entity's neighbors. A tensor
        is accepted as a parameter for copying purposes.

        Parameters
        ----------
        worlds : ``List[SpiderWorld]``
        num_entities : ``int``
        tensor : ``torch.Tensor``
            Used for copying the constructed list onto the right device.

        Returns
        -------
        A ``torch.LongTensor`` with shape ``(batch_size, num_entities, num_neighbors)``. It is padded
        with -1 instead of 0, since 0 is a valid neighbor index. If all the entities in the batch
        have no neighbors, None will be returned.
        """

        num_neighbors = 0
        for world in worlds:
            for entity in world.db_context.knowledge_graph.entities:
                if len(world.db_context.knowledge_graph.neighbors[entity]) > num_neighbors:
                    num_neighbors = len(world.db_context.knowledge_graph.neighbors[entity])

        batch_neighbors = []
        no_entities_have_neighbors = True
        for world in worlds:
            # Each batch instance has its own world, which has a corresponding table.
            entities = world.db_context.knowledge_graph.entities
            entity2index = {entity: i for i, entity in enumerate(entities)}
            entity2neighbors = world.db_context.knowledge_graph.neighbors
            neighbor_indexes = []
            for entity in entities:
                entity_neighbors = [entity2index[n] for n in entity2neighbors[entity]]
                if entity_neighbors:
                    no_entities_have_neighbors = False
                # Pad with -1 instead of 0, since 0 represents a neighbor index.
                padded = pad_sequence_to_length(entity_neighbors, num_neighbors, lambda: -1)
                neighbor_indexes.append(padded)
            neighbor_indexes = pad_sequence_to_length(neighbor_indexes,
                                                      num_entities,
                                                      lambda: [-1] * num_neighbors)
            batch_neighbors.append(neighbor_indexes)
        # It is possible that none of the entities has any neighbors, since our definition of the
        # knowledge graph allows it when no entities or numbers were extracted from the question.
        if no_entities_have_neighbors:
            return None
        return torch.tensor(batch_neighbors, device=device, dtype=torch.long)

    def _get_schema_graph_encoding(self,
                                   worlds: List[SpiderWorld],
                                   initial_graph_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        max_num_entities = max([len(world.db_context.knowledge_graph.entities) for world in worlds])
        batch_size = initial_graph_embeddings.size(0)

        graph_data_list = []

        for batch_index, world in enumerate(worlds):
            x = initial_graph_embeddings[batch_index]

            adj_list = self._get_graph_adj_lists(initial_graph_embeddings.device,
                                                 world, initial_graph_embeddings.size(1) - 1)
            graph_data = Data(x)
            for i, l in enumerate(adj_list):
                graph_data[f'edge_index_{i}'] = l
            graph_data_list.append(graph_data)

        batch = Batch.from_data_list(graph_data_list)

        gnn_output = self._gnn(batch.x, [batch[f'edge_index_{i}'] for i in range(self._gnn.num_edge_types)])

        num_nodes = max_num_entities
        gnn_output = gnn_output.view(batch_size, num_nodes, -1)
        # entities_encodings = gnn_output
        entities_encodings = gnn_output[:, :max_num_entities]
        # global_node_encodings = gnn_output[:, max_num_entities]

        return entities_encodings

    @staticmethod
    def _get_graph_adj_lists(device, world, global_entity_id, global_node=False):
        entity_mapping = {}
        for i, entity in enumerate(world.db_context.knowledge_graph.entities):
            entity_mapping[entity] = i
        entity_mapping['_global_'] = global_entity_id
        adj_list_own = []  # column--table
        adj_list_link = []  # table->table / foreign->primary
        adj_list_linked = []  # table<-table / foreign<-primary
        adj_list_global = []  # node->global

        # TODO: Prepare in advance?
        for key, neighbors in world.db_context.knowledge_graph.neighbors.items():
            idx_source = entity_mapping[key]
            for n_key in neighbors:
                idx_target = entity_mapping[n_key]
                if n_key.startswith("table") or key.startswith("table"):
                    adj_list_own.append((idx_source, idx_target))
                elif n_key.startswith("string") or key.startswith("string"):
                    adj_list_own.append((idx_source, idx_target))
                elif key.startswith("column:foreign"):
                    adj_list_link.append((idx_source, idx_target))
                    src_table_key = f"table:{key.split(':')[2]}"
                    tgt_table_key = f"table:{n_key.split(':')[2]}"
                    idx_source_table = entity_mapping[src_table_key]
                    idx_target_table = entity_mapping[tgt_table_key]
                    adj_list_link.append((idx_source_table, idx_target_table))
                elif n_key.startswith("column:foreign"):
                    adj_list_linked.append((idx_source, idx_target))
                    src_table_key = f"table:{key.split(':')[2]}"
                    tgt_table_key = f"table:{n_key.split(':')[2]}"
                    idx_source_table = entity_mapping[src_table_key]
                    idx_target_table = entity_mapping[tgt_table_key]
                    adj_list_linked.append((idx_source_table, idx_target_table))
                else:
                    assert False

            adj_list_global.append((idx_source, entity_mapping['_global_']))

        all_adj_types = [adj_list_own, adj_list_link, adj_list_linked]

        if global_node:
            all_adj_types.append(adj_list_global)

        return [torch.tensor(l, device=device, dtype=torch.long).transpose(0, 1) if l
                else torch.tensor(l, device=device, dtype=torch.long)
                for l in all_adj_types]

    def _create_grammar_state(self,
                              world: SpiderWorld,
                              possible_actions: List[ProductionRule],
                              linking_scores: torch.Tensor,
                              linked_actions_linking_scores: torch.Tensor,
                              entity_types: torch.Tensor,
                              entity_graph_encoding: torch.Tensor) -> GrammarStatelet:
        action_map = {}
        for action_index, action in enumerate(possible_actions):
            action_string = action[0]
            action_map[action_string] = action_index

        valid_actions = world.valid_actions
        entity_map = {}
        entities = world.entities_names

        for entity_index, entity in enumerate(entities):
            entity_map[entity] = entity_index

        translated_valid_actions: Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor, List[int]]]] = {}
        for key, action_strings in valid_actions.items():
            translated_valid_actions[key] = {}
            # `key` here is a non-terminal from the grammar, and `action_strings` are all the valid
            # productions of that non-terminal.  We'll first split those productions by global vs.
            # linked action.

            action_indices = [action_map[action_string] for action_string in action_strings]
            production_rule_arrays = [(possible_actions[index], index) for index in action_indices]
            global_actions = []
            linked_actions = []
            for production_rule_array, action_index in production_rule_arrays:
                if production_rule_array[1]:
                    global_actions.append((production_rule_array[2], action_index))
                else:
                    linked_actions.append((production_rule_array[0], action_index))

            if global_actions:
                global_action_tensors, global_action_ids = zip(*global_actions)
                global_action_tensor = torch.cat(global_action_tensors, dim=0).to(
                    global_action_tensors[0].device).long()
                global_input_embeddings = self._action_embedder(global_action_tensor)
                global_output_embeddings = self._output_action_embedder(global_action_tensor)
                translated_valid_actions[key]['global'] = (global_input_embeddings,
                                                           global_output_embeddings,
                                                           list(global_action_ids))
            if linked_actions:
                linked_rules, linked_action_ids = zip(*linked_actions)
                entities = [rule.split(' -> ')[1].strip('[]\"') for rule in linked_rules]

                entity_ids = [entity_map[entity] for entity in entities]

                entity_linking_scores = linking_scores[entity_ids]

                if linked_actions_linking_scores is not None:
                    entity_action_linking_scores = linked_actions_linking_scores[entity_ids]

                if not self._decoder_use_graph_entities:
                    entity_type_tensor = entity_types[entity_ids]
                    entity_type_embeddings = (self._entity_type_decoder_embedding(entity_type_tensor)
                                              .to(entity_types.device)
                                              .float())
                else:
                    entity_type_embeddings = entity_graph_encoding.index_select(
                        dim=0,
                        index=torch.tensor(entity_ids, device=entity_graph_encoding.device)
                    )

                if self._self_attend:
                    translated_valid_actions[key]['linked'] = (entity_linking_scores,
                                                               entity_type_embeddings,
                                                               list(linked_action_ids),
                                                               entity_action_linking_scores)
                else:
                    translated_valid_actions[key]['linked'] = (entity_linking_scores,
                                                               entity_type_embeddings,
                                                               list(linked_action_ids))

        return GrammarStatelet(['statement'],
                               translated_valid_actions,
                               self.is_nonterminal)

    @staticmethod
    def is_nonterminal(token: str):
        if token[0] == '"' and token[-1] == '"':
            return False
        return True

    def _get_linking_probabilities(self,
                                   worlds: List[SpiderWorld],
                                   linking_scores: torch.FloatTensor,
                                   question_mask: torch.LongTensor,
                                   entity_type_dict: Dict[int, int]) -> torch.FloatTensor:
        """
        Produces the probability of an entity given a question word and type. The logic below
        separates the entities by type since the softmax normalization term sums over entities
        of a single type.

        Parameters
        ----------
        worlds : ``List[WikiTablesWorld]``
        linking_scores : ``torch.FloatTensor``
            Has shape (batch_size, num_question_tokens, num_entities).
        question_mask: ``torch.LongTensor``
            Has shape (batch_size, num_question_tokens).
        entity_type_dict : ``Dict[int, int]``
            This is a mapping from ((batch_index * num_entities) + entity_index) to entity type id.

        Returns
        -------
        batch_probabilities : ``torch.FloatTensor``
            Has shape ``(batch_size, num_question_tokens, num_entities)``.
            Contains all the probabilities for an entity given a question word.
        """
        _, num_question_tokens, num_entities = linking_scores.size()
        batch_probabilities = []

        for batch_index, world in enumerate(worlds):
            all_probabilities = []
            num_entities_in_instance = 0

            # NOTE: The way that we're doing this here relies on the fact that entities are
            # implicitly sorted by their types when we sort them by name, and that numbers come
            # before "date_column:", followed by "number_column:", "string:", and "string_column:".
            # This is not a great assumption, and could easily break later, but it should work for now.
            for type_index in range(self._num_entity_types):
                # This index of 0 is for the null entity for each type, representing the case where a
                # word doesn't link to any entity.
                entity_indices = [0]
                entities = world.db_context.knowledge_graph.entities
                for entity_index, _ in enumerate(entities):
                    if entity_type_dict[batch_index * num_entities + entity_index] == type_index:
                        entity_indices.append(entity_index)

                if len(entity_indices) == 1:
                    # No entities of this type; move along...
                    continue

                # We're subtracting one here because of the null entity we added above.
                num_entities_in_instance += len(entity_indices) - 1

                # We separate the scores by type, since normalization is done per type.  There's an
                # extra "null" entity per type, also, so we have `num_entities_per_type + 1`.  We're
                # selecting from a (num_question_tokens, num_entities) linking tensor on _dimension 1_,
                # so we get back something of shape (num_question_tokens,) for each index we're
                # selecting.  All of the selected indices together then make a tensor of shape
                # (num_question_tokens, num_entities_per_type + 1).
                indices = linking_scores.new_tensor(entity_indices, dtype=torch.long)
                entity_scores = linking_scores[batch_index].index_select(1, indices)

                # We used index 0 for the null entity, so this will actually have some values in it.
                # But we want the null entity's score to be 0, so we set that here.
                entity_scores[:, 0] = 0

                # No need for a mask here, as this is done per batch instance, with no padding.
                type_probabilities = torch.nn.functional.softmax(entity_scores, dim=1)
                all_probabilities.append(type_probabilities[:, 1:])

            # We need to add padding here if we don't have the right number of entities.
            if num_entities_in_instance != num_entities:
                zeros = linking_scores.new_zeros(num_question_tokens,
                                                 num_entities - num_entities_in_instance)
                all_probabilities.append(zeros)

            # (num_question_tokens, num_entities)
            probabilities = torch.cat(all_probabilities, dim=1)
            batch_probabilities.append(probabilities)
        batch_probabilities = torch.stack(batch_probabilities, dim=0)
        return batch_probabilities * question_mask.unsqueeze(-1).float()

    @staticmethod
    def _action_history_match(predicted: List[int], targets: torch.LongTensor) -> int:
        # TODO(mattg): this could probably be moved into a FullSequenceMatch metric, or something.
        # Check if target is big enough to cover prediction (including start/end symbols)
        if len(predicted) > targets.size(0):
            return 0
        predicted_tensor = targets.new_tensor(predicted)
        targets_trimmed = targets[:len(predicted)]
        # Return 1 if the predicted sequence is anywhere in the list of targets.
        return torch.max(torch.min(targets_trimmed.eq(predicted_tensor), dim=0)[0]).item()

    @staticmethod
    def _query_difficulty(targets: torch.LongTensor, action_mapping, batch_index):
        number_tables = len([action_mapping[(batch_index, int(a))] for a in targets if
                             a >= 0 and action_mapping[(batch_index, int(a))].startswith('table_name')])
        return number_tables > 1

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            '_match/exact_match': self._exact_match.get_metric(reset),
            'sql_match': self._sql_evaluator_match.get_metric(reset),
            '_others/action_similarity': self._action_similarity.get_metric(reset),
            '_match/match_single': self._acc_single.get_metric(reset),
            '_match/match_hard': self._acc_multi.get_metric(reset),
            'beam_hit': self._beam_hit.get_metric(reset)
        }

    @staticmethod
    def _get_type_vector(worlds: List[SpiderWorld],
                         num_entities: int,
                         device) -> Tuple[torch.LongTensor, Dict[int, int]]:
        """
        Produces the encoding for each entity's type. In addition, a map from a flattened entity
        index to type is returned to combine entity type operations into one method.

        Parameters
        ----------
        worlds : ``List[AtisWorld]``
        num_entities : ``int``
        tensor : ``torch.Tensor``
            Used for copying the constructed list onto the right device.

        Returns
        -------
        A ``torch.LongTensor`` with shape ``(batch_size, num_entities, num_types)``.
        entity_types : ``Dict[int, int]``
            This is a mapping from ((batch_index * num_entities) + entity_index) to entity type id.
        """
        entity_types = {}
        batch_types = []

        column_type_ids = ['boolean', 'foreign', 'number', 'others', 'primary', 'text', 'time']

        for batch_index, world in enumerate(worlds):
            types = []

            for entity_index, entity in enumerate(world.db_context.knowledge_graph.entities):
                parts = entity.split(':')
                entity_main_type = parts[0]
                if entity_main_type == 'column':
                    column_type = parts[1]
                    entity_type = column_type_ids.index(column_type)
                elif entity_main_type == 'string':
                    # cell value
                    entity_type = len(column_type_ids)
                elif entity_main_type == 'table':
                    entity_type = len(column_type_ids) + 1
                else:
                    raise (Exception("Unkown entity"))
                types.append(entity_type)

                # For easier lookups later, we're actually using a _flattened_ version
                # of (batch_index, entity_index) for the key, because this is how the
                # linking scores are stored.
                flattened_entity_index = batch_index * num_entities + entity_index
                entity_types[flattened_entity_index] = entity_type
            padded = pad_sequence_to_length(types, num_entities, lambda: 0)
            batch_types.append(padded)

        return torch.tensor(batch_types, dtype=torch.long, device=device), entity_types

    def _compute_validation_outputs(self,
                                    actions: List[List[ProductionRuleArray]],
                                    best_final_states: Mapping[int, Sequence[GrammarBasedState]],
                                    world: List[SpiderWorld],
                                    target_list: List[List[str]],
                                    outputs: Dict[str, Any]) -> None:
        batch_size = len(actions)

        outputs['predicted_sql_query'] = []

        action_mapping = {}
        for batch_index, batch_actions in enumerate(actions):
            for action_index, action in enumerate(batch_actions):
                action_mapping[(batch_index, action_index)] = action[0]

        for i in range(batch_size):
            # gold sql exactly as given
            original_gold_sql_query = ' '.join(world[i].get_query_without_table_hints())

            if i not in best_final_states:
                self._exact_match(0)
                self._action_similarity(0)
                self._sql_evaluator_match(0)
                self._acc_multi(0)
                self._acc_single(0)
                outputs['predicted_sql_query'].append('')
                continue

            best_action_indices = best_final_states[i][0].action_history[0]

            action_strings = [action_mapping[(i, action_index)]
                              for action_index in best_action_indices]
            predicted_sql_query = action_sequence_to_sql(action_strings, add_table_names=True)
            outputs['predicted_sql_query'].append(sqlparse.format(predicted_sql_query, reindent=False))

            if target_list is not None:
                targets = target_list[i].data

                sequence_in_targets = self._action_history_match(best_action_indices, targets)
                self._exact_match(sequence_in_targets)

                sql_evaluator_match = self._evaluate_func(original_gold_sql_query, predicted_sql_query, world[i].db_id)
                self._sql_evaluator_match(sql_evaluator_match)

                similarity = difflib.SequenceMatcher(None, best_action_indices, targets)
                self._action_similarity(similarity.ratio())

                difficulty = self._query_difficulty(targets, action_mapping, i)
                if difficulty:
                    self._acc_multi(sql_evaluator_match)
                else:
                    self._acc_single(sql_evaluator_match)

            beam_hit = False
            for pos, final_state in enumerate(best_final_states[i]):
                action_indices = final_state.action_history[0]
                action_strings = [action_mapping[(i, action_index)]
                                  for action_index in action_indices]
                candidate_sql_query = action_sequence_to_sql(action_strings, add_table_names=True)

                if target_list is not None:
                    correct = self._evaluate_func(original_gold_sql_query, candidate_sql_query, world[i].db_id)
                    if correct:
                        beam_hit = True
                    self._beam_hit(beam_hit)
