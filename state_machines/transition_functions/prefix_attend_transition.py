from typing import Dict, Tuple, List, Set, Any, Callable

import torch
from allennlp.modules import Attention, FeedForward
from allennlp.nn import Activation, util

from allennlp.state_machines.states.grammar_based_state import GrammarBasedState
from allennlp.state_machines.transition_functions.linking_transition_function import LinkingTransitionFunction
from overrides import overrides
from torch.nn import Linear

from state_machines.states.rnn_statelet import RnnStatelet


class PrefixAttendTransitionFunction(LinkingTransitionFunction):
    def __init__(self,
                 encoder_output_dim: int,
                 action_embedding_dim: int,
                 input_attention: Attention,
                 output_attention: Attention,
                 activation: Activation = Activation.by_name('relu')(),
                 predict_start_type_separately: bool = True,
                 num_start_types: int = None,
                 add_action_bias: bool = True,
                 mixture_feedforward: FeedForward = None,
                 dropout: float = 0.0,
                 num_layers: int = 1) -> None:
        super().__init__(encoder_output_dim=encoder_output_dim,
                         action_embedding_dim=action_embedding_dim,
                         input_attention=input_attention,
                         num_start_types=num_start_types,
                         activation=activation,
                         predict_start_type_separately=predict_start_type_separately,
                         add_action_bias=add_action_bias,
                         dropout=dropout,
                         num_layers=num_layers,
                         mixture_feedforward=mixture_feedforward)
        self._output_attention = output_attention

        # override
        self._input_projection_layer = Linear(encoder_output_dim + action_embedding_dim, encoder_output_dim)

        self._attend_output_projection_layer = Linear(encoder_output_dim*2, encoder_output_dim)

        self._first_attended_output = torch.nn.Parameter(torch.FloatTensor(action_embedding_dim))
        torch.nn.init.normal_(self._first_attended_output)

    @overrides
    def take_step(self,
                  state: GrammarBasedState,
                  max_actions: int = None,
                  allowed_actions: List[Set[int]] = None) -> List[GrammarBasedState]:
        if self._predict_start_type_separately and not state.action_history[0]:
            # The wikitables parser did something different when predicting the start type, which
            # is our first action.  So in this case we break out into a different function.  We'll
            # ignore max_actions on our first step, assuming there aren't that many start types.
            return self._take_first_step(state, allowed_actions)

        # Taking a step in the decoder consists of three main parts.  First, we'll construct the
        # input to the decoder and update the decoder's hidden state.  Second, we'll use this new
        # hidden state (and maybe other information) to predict an action.  Finally, we will
        # construct new states for the next step.  Each new state corresponds to one valid action
        # that can be taken from the current state, and they are ordered by their probability of
        # being selected.

        updated_state = self._update_decoder_state(state)
        batch_results = self._compute_action_probabilities(state,
                                                           updated_state['hidden_state'],
                                                           updated_state['attention_weights'],
                                                           updated_state['predicted_action_embeddings'])
        new_states = self._construct_next_states(state,
                                                 updated_state,
                                                 batch_results,
                                                 max_actions,
                                                 allowed_actions)

        return new_states

    @overrides
    def _update_decoder_state(self, state: GrammarBasedState) -> Dict[str, torch.Tensor]:
        # For updating the decoder, we're doing a bunch of tensor operations that can be batched
        # without much difficulty.  So, we take all group elements and batch their tensors together
        # before doing these decoder operations.

        group_size = len(state.batch_indices)
        attended_question = torch.stack([rnn_state.attended_input for rnn_state in state.rnn_state])

        hidden_state = torch.stack([rnn_state.hidden_state for rnn_state in state.rnn_state])
        memory_cell = torch.stack([rnn_state.memory_cell for rnn_state in state.rnn_state])

        previous_action_embedding = torch.stack([rnn_state.previous_action_embedding
                                                 for rnn_state in state.rnn_state])

        if not state.action_history[0]:
            attended_output, output_attention_weights = self._first_attended_output.unsqueeze(0).repeat(group_size, 1),\
                                                        None
        else:
            decoder_outputs = torch.stack([rnn_state.decoder_outputs for rnn_state in state.rnn_state])
            decoded_item_embeddings = torch.stack([rnn_state.decoded_item_embeddings for rnn_state in state.rnn_state])

            action_query = torch.cat([hidden_state, attended_question], dim=-1)

            # (group_size, action_embedding_dim)
            projected_query = self._activation(self._attend_output_projection_layer(action_query))
            query = self._dropout(projected_query)

            attended_output, output_attention_weights = self.attend(query,
                                                                    decoder_outputs,
                                                                    decoded_item_embeddings)

        # (group_size, decoder_input_dim)
        projected_input = self._input_projection_layer(torch.cat([attended_question,
                                                                  previous_action_embedding + attended_output], -1))
        decoder_input = self._activation(projected_input)
        hidden_state, memory_cell = self._decoder_cell(decoder_input, (hidden_state, memory_cell))
        hidden_state = self._dropout(hidden_state)

        # (group_size, encoder_output_dim)
        encoder_outputs = torch.stack([state.rnn_state[0].encoder_outputs[i] for i in state.batch_indices])
        encoder_output_mask = torch.stack([state.rnn_state[0].encoder_output_mask[i] for i in state.batch_indices])

        attended_question, attention_weights = self.attend_on_question(hidden_state,
                                                                       encoder_outputs,
                                                                       encoder_output_mask)
        action_query = torch.cat([hidden_state, attended_question], dim=-1)

        # (group_size, action_embedding_dim)
        projected_query = self._activation(self._output_projection_layer(action_query))
        predicted_action_embeddings = self._dropout(projected_query)
        if self._add_action_bias:
            # NOTE: It's important that this happens right before the dot product with the action
            # embeddings.  Otherwise this isn't a proper bias.  We do it here instead of right next
            # to the `.mm` below just so we only do it once for the whole group.
            ones = predicted_action_embeddings.new([[1] for _ in range(group_size)])
            predicted_action_embeddings = torch.cat([predicted_action_embeddings, ones], dim=-1)
        return {
            'hidden_state': hidden_state,
            'memory_cell': memory_cell,
            'attended_question': attended_question,
            'attended_output': attended_output,
            'attention_weights': attention_weights,
            'output_attention_weights': output_attention_weights,
            'predicted_action_embeddings': predicted_action_embeddings
        }

    def attend(self,
               query: torch.Tensor,
               key: torch.Tensor,
               value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given a query (which is typically the decoder hidden state), compute an attention over the
        output of the question encoder, and return a weighted sum of the question representations
        given this attention.  We also return the attention weights themselves.

        This is a simple computation, but we have it as a separate method so that the ``forward``
        method on the main parser module can call it on the initial hidden state, to simplify the
        logic in ``take_step``.
        """
        # (group_size, question_length)
        question_attention_weights = self._output_attention(query, key, None)

        # (group_size, encoder_output_dim)
        attended_question = util.weighted_sum(value, question_attention_weights)
        return attended_question, question_attention_weights

    def _construct_next_states(self,
                               state: GrammarBasedState,
                               updated_rnn_state: Dict[str, torch.Tensor],
                               batch_action_probs: Dict[int, List[Tuple[int, Any, Any, Any, List[int]]]],
                               max_actions: int,
                               allowed_actions: List[Set[int]]):
        # pylint: disable=no-self-use

        # We'll yield a bunch of states here that all have a `group_size` of 1, so that the
        # learning algorithm can decide how many of these it wants to keep, and it can just regroup
        # them later, as that's a really easy operation.
        #
        # We first define a `make_state` method, as in the logic that follows we want to create
        # states in a couple of different branches, and we don't want to duplicate the
        # state-creation logic.  This method creates a closure using variables from the method, so
        # it doesn't make sense to pull it out of here.

        # Each group index here might get accessed multiple times, and doing the slicing operation
        # each time is more expensive than doing it once upfront.  These three lines give about a
        # 10% speedup in training time.
        group_size = len(state.batch_indices)

        chunk_index = 1 if self._num_layers > 1 else 0
        hidden_state = [x.squeeze(chunk_index)
                        for x in updated_rnn_state['hidden_state'].chunk(group_size, chunk_index)]
        memory_cell = [x.squeeze(chunk_index)
                       for x in updated_rnn_state['memory_cell'].chunk(group_size, chunk_index)]

        if not state.action_history[0]:
            decoder_outputs = updated_rnn_state['hidden_state'].unsqueeze(1)
        else:
            decoder_outputs = torch.cat((
                torch.stack([x.decoder_outputs for x in state.rnn_state]),
                updated_rnn_state['hidden_state'].unsqueeze(1)
            ), dim=1)

        attended_question = [x.squeeze(0) for x in updated_rnn_state['attended_question'].chunk(group_size, 0)]

        def make_state(group_index: int,
                       action: int,
                       new_score: torch.Tensor,
                       action_embedding: torch.Tensor) -> GrammarBasedState:
            batch_index = state.batch_indices[group_index]
            action_entity_id = state.action_entity_mapping[batch_index][action] + 1  # add 1 so that -1 becomes 0 (pad)
            if not state.action_history[0]:
                decoded_item_embeddings = state.rnn_state[group_index].item_embeddings[action_entity_id].unsqueeze(0)
            else:
                decoded_item_embeddings = torch.cat((
                    state.rnn_state[group_index].decoded_item_embeddings,
                    state.rnn_state[group_index].item_embeddings[action_entity_id].unsqueeze(0)
                ), dim=0)

            new_rnn_state = RnnStatelet(hidden_state[group_index],
                                        memory_cell[group_index],
                                        action_embedding,
                                        attended_question[group_index],
                                        state.rnn_state[group_index].encoder_outputs,
                                        state.rnn_state[group_index].encoder_output_mask,
                                        state.rnn_state[group_index].item_embeddings,
                                        decoder_outputs[group_index],
                                        decoded_item_embeddings)
            for i, _, current_log_probs, _, actions in batch_action_probs[batch_index]:
                if i == group_index:
                    considered_actions = actions
                    probabilities = current_log_probs.exp().cpu()
                    break
            return state.new_state_from_group_index(group_index,
                                                    action,
                                                    new_score,
                                                    new_rnn_state,
                                                    considered_actions,
                                                    probabilities,
                                                    updated_rnn_state['attention_weights'],
                                                    updated_rnn_state['output_attention_weights'])

        new_states = []
        for _, results in batch_action_probs.items():
            if allowed_actions and not max_actions:
                # If we're given a set of allowed actions, and we're not just keeping the top k of
                # them, we don't need to do any sorting, so we can speed things up quite a bit.
                for group_index, log_probs, _, action_embeddings, actions in results:
                    for log_prob, action_embedding, action in zip(log_probs, action_embeddings, actions):
                        if action in allowed_actions[group_index]:
                            new_states.append(make_state(group_index, action, log_prob, action_embedding))
            else:
                # In this case, we need to sort the actions.  We'll do that on CPU, as it's easier,
                # and our action list is on the CPU, anyway.
                group_indices = []
                group_log_probs: List[torch.Tensor] = []
                group_action_embeddings = []
                group_actions = []
                for group_index, log_probs, _, action_embeddings, actions in results:
                    group_indices.extend([group_index] * len(actions))
                    group_log_probs.append(log_probs)
                    group_action_embeddings.append(action_embeddings)
                    group_actions.extend(actions)
                log_probs = torch.cat(group_log_probs, dim=0)
                action_embeddings = torch.cat(group_action_embeddings, dim=0)
                log_probs_cpu = log_probs.data.cpu().numpy().tolist()
                batch_states = [(log_probs_cpu[i],
                                 group_indices[i],
                                 log_probs[i],
                                 action_embeddings[i],
                                 group_actions[i])
                                for i in range(len(group_actions))
                                if (not allowed_actions or
                                    group_actions[i] in allowed_actions[group_indices[i]])]
                # We use a key here to make sure we're not trying to compare anything on the GPU.
                batch_states.sort(key=lambda x: x[0], reverse=True)
                if max_actions:
                    batch_states = batch_states[:max_actions]
                for _, group_index, log_prob, action_embedding, action in batch_states:
                    new_states.append(make_state(group_index, action, log_prob, action_embedding))
        return new_states
