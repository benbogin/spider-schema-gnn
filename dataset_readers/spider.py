import json
import logging
import os
from typing import List, Dict

import dill
from allennlp.common.checks import ConfigurationError
from allennlp.data import DatasetReader, Tokenizer, TokenIndexer, Field, Instance
from allennlp.data.fields import TextField, ProductionRuleField, ListField, IndexField, MetadataField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from overrides import overrides
from spacy.symbols import ORTH, LEMMA

from dataset_readers.dataset_util.spider_utils import fix_number_value, disambiguate_items
from dataset_readers.fields.knowledge_graph_field import SpiderKnowledgeGraphField
from semparse.contexts.spider_db_context import SpiderDBContext
from semparse.worlds.spider_world import SpiderWorld

logger = logging.getLogger(__name__)


@DatasetReader.register("spider")
class SpiderDatasetReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 question_token_indexers: Dict[str, TokenIndexer] = None,
                 keep_if_unparsable: bool = True,
                 tables_file: str = None,
                 dataset_path: str = 'dataset/database',
                 load_cache: bool = True,
                 save_cache: bool = True,
                 loading_limit = -1):
        super().__init__(lazy=lazy)

        # default spacy tokenizer splits the common token 'id' to ['i', 'd'], we here write a manual fix for that
        spacy_tokenizer = SpacyWordSplitter(pos_tags=True)
        spacy_tokenizer.spacy.tokenizer.add_special_case(u'id', [{ORTH: u'id', LEMMA: u'id'}])
        self._tokenizer = WordTokenizer(spacy_tokenizer)

        self._utterance_token_indexers = question_token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._keep_if_unparsable = keep_if_unparsable

        self._tables_file = tables_file
        self._dataset_path = dataset_path

        self._load_cache = load_cache
        self._save_cache = save_cache
        self._loading_limit = loading_limit

    @overrides
    def _read(self, file_path: str):
        if not file_path.endswith('.json'):
            raise ConfigurationError(f"Don't know how to read filetype of {file_path}")

        cache_dir = os.path.join('cache', file_path.split("/")[-1])

        if self._load_cache:
            logger.info(f'Trying to load cache from {cache_dir}')
        if self._save_cache:
            os.makedirs(cache_dir, exist_ok=True)

        cnt = 0
        with open(file_path, "r") as data_file:
            json_obj = json.load(data_file)
            for total_cnt, ex in enumerate(json_obj):
                cache_filename = f'instance-{total_cnt}.pt'
                cache_filepath = os.path.join(cache_dir, cache_filename)
                if self._loading_limit == cnt:
                    break

                if self._load_cache:
                    try:
                        ins = dill.load(open(cache_filepath, 'rb'))
                        if ins is None and not self._keep_if_unparsable:
                            # skip unparsed examples
                            continue
                        yield ins
                        cnt += 1
                        continue
                    except Exception as e:
                        # could not load from cache - keep loading without cache
                        pass

                query_tokens = None
                if 'query_toks' in ex:
                    # we only have 'query_toks' in example for training/dev sets

                    # fix for examples: we want to use the 'query_toks_no_value' field of the example which anonymizes
                    # values. However, it also anonymizes numbers (e.g. LIMIT 3 -> LIMIT 'value', which is not good
                    # since the official evaluator does expect a number and not a value
                    ex = fix_number_value(ex)

                    # we want the query tokens to be non-ambiguous (i.e. know for each column the table it belongs to,
                    # and for each table alias its explicit name)
                    # we thus remove all aliases and make changes such as:
                    # 'name' -> 'singer@name',
                    # 'singer AS T1' -> 'singer',
                    # 'T1.name' -> 'singer@name'
                    try:
                        query_tokens = disambiguate_items(ex['db_id'], ex['query_toks_no_value'],
                                                                self._tables_file, allow_aliases=False)
                    except Exception as e:
                        # there are two examples in the train set that are wrongly formatted, skip them
                        print(f"error with {ex['query']}")
                        print(e)

                ins = self.text_to_instance(
                    utterance=ex['question'],
                    db_id=ex['db_id'],
                    sql=query_tokens)
                if ins is not None:
                    cnt += 1
                if self._save_cache:
                    dill.dump(ins, open(cache_filepath, 'wb'))

                if ins is not None:
                    yield ins

    def text_to_instance(self,
                         utterance: str,
                         db_id: str,
                         sql: List[str] = None):
        fields: Dict[str, Field] = {}

        db_context = SpiderDBContext(db_id, utterance, tokenizer=self._tokenizer,
                                     tables_file=self._tables_file, dataset_path=self._dataset_path)
        table_field = SpiderKnowledgeGraphField(db_context.knowledge_graph,
                                                db_context.tokenized_utterance,
                                                self._utterance_token_indexers,
                                                entity_tokens=db_context.entity_tokens,
                                                include_in_vocab=False,  # TODO: self._use_table_for_vocab,
                                                max_table_tokens=None)  # self._max_table_tokens)

        world = SpiderWorld(db_context, query=sql)
        fields["utterance"] = TextField(db_context.tokenized_utterance, self._utterance_token_indexers)

        action_sequence, all_actions = world.get_action_sequence_and_all_actions()

        if action_sequence is None and self._keep_if_unparsable:
            # print("Parse error")
            action_sequence = []
        elif action_sequence is None:
            return None

        index_fields: List[Field] = []
        production_rule_fields: List[Field] = []

        for production_rule in all_actions:
            nonterminal, rhs = production_rule.split(' -> ')
            production_rule = ' '.join(production_rule.split(' '))
            field = ProductionRuleField(production_rule,
                                        world.is_global_rule(rhs),
                                        nonterminal=nonterminal)
            production_rule_fields.append(field)

        valid_actions_field = ListField(production_rule_fields)
        fields["valid_actions"] = valid_actions_field

        action_map = {action.rule: i  # type: ignore
                      for i, action in enumerate(valid_actions_field.field_list)}

        for production_rule in action_sequence:
            index_fields.append(IndexField(action_map[production_rule], valid_actions_field))
        if not action_sequence:
            index_fields = [IndexField(-1, valid_actions_field)]

        action_sequence_field = ListField(index_fields)
        fields["action_sequence"] = action_sequence_field
        fields["world"] = MetadataField(world)
        fields["schema"] = table_field
        return Instance(fields)
