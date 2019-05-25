from typing import List, Tuple, Dict, Set, Optional
from copy import deepcopy

from parsimonious import Grammar
from parsimonious.exceptions import ParseError

from semparse.contexts.spider_context_utils import format_grammar_string, initialize_valid_actions, SqlVisitor
from semparse.contexts.spider_db_context import SpiderDBContext
from semparse.contexts.spider_db_grammar import GRAMMAR_DICTIONARY, update_grammar_with_tables, \
    update_grammar_to_be_table_names_free, update_grammar_flip_joins


class SpiderWorld:
    """
    World representation for spider dataset.
    """

    def __init__(self, db_context: SpiderDBContext, query: Optional[List[str]], allow_alias: bool = False) -> None:
        self.db_id = db_context.db_id
        self.allow_alias = allow_alias

        # NOTE: This base dictionary should not be modified.
        self.base_grammar_dictionary = deepcopy(GRAMMAR_DICTIONARY)
        self.query = query
        self.db_context = db_context

        # keep a list of entities names as they are given in sql queries
        self.entities_names = {}
        for i, entity in enumerate(self.db_context.knowledge_graph.entities):
            parts = entity.split(':')
            if parts[0] in ['table', 'string']:
                self.entities_names[parts[1]] = i
            else:
                _, _, table_name, column_name = parts
                self.entities_names[f'{table_name}@{column_name}'] = i
        self.valid_actions = []
        self.valid_actions_flat = []

    def get_action_sequence_and_all_actions(self,
                                            allow_aliases: bool = False) -> Tuple[List[str], List[str]]:
        grammar_with_context = deepcopy(self.base_grammar_dictionary)
        if not allow_aliases:
            update_grammar_to_be_table_names_free(grammar_with_context)

        schema = self.db_context.schema

        update_grammar_with_tables(grammar_with_context, schema)
        grammar = Grammar(format_grammar_string(grammar_with_context))

        valid_actions = initialize_valid_actions(grammar)
        all_actions = set()
        for action_list in valid_actions.values():
            all_actions.update(action_list)
        sorted_actions = sorted(all_actions)
        self.valid_actions = valid_actions
        self.valid_actions_flat = sorted_actions

        action_sequence = None
        if self.query is not None:
            sql_visitor = SqlVisitor(grammar)
            query = " ".join(self.query).lower().replace("``", "'").replace("''", "'")
            try:
                action_sequence = sql_visitor.parse(query) if query else []
            except ParseError as e:
                pass

        return action_sequence, sorted_actions

    def get_all_actions(self, schema,
                        flip_joins: bool,
                        allow_aliases: bool) -> Tuple[List[str], List[str]]:
        grammar_with_context = deepcopy(self.base_grammar_dictionary)
        if not allow_aliases:
            update_grammar_to_be_table_names_free(grammar_with_context)

        if flip_joins:
            update_grammar_flip_joins(grammar_with_context)

        update_grammar_with_tables(grammar_with_context, schema, self.db_id)
        grammar = Grammar(format_grammar_string(grammar_with_context))

        valid_actions = initialize_valid_actions(grammar)
        all_actions = set()
        for action_list in valid_actions.values():
            all_actions.update(action_list)
        sorted_actions = sorted(all_actions)
        self.valid_actions = valid_actions
        self.valid_actions_flat = sorted_actions

        return sorted_actions

    def is_global_rule(self, rhs: str) -> bool:
        rhs = rhs.strip('[] ')
        if rhs[0] != '"':
            return True
        return rhs.strip('"') not in self.entities_names

    def get_oracle_relevance_score(self, oracle_entities: set):
        """
        return 0/1 for each schema item if it should be in the graph,
        given the used entities in the gold answer
        """
        scores = [0 for _ in range(len(self.db_context.knowledge_graph.entities))]

        for i, entity in enumerate(self.db_context.knowledge_graph.entities):
            parts = entity.split(':')
            if parts[0] == 'column':
                name = parts[2] + '@' + parts[3]
            else:
                name = parts[-1]
            if name in oracle_entities:
                scores[i] = 1

        return scores

    def get_action_entity_mapping(self) -> Dict[int, int]:
        mapping = {}

        for action_index, action in enumerate(self.valid_actions_flat):
            # default is padding
            mapping[action_index] = -1

            action = action.split(" -> ")[1].strip('[]')
            action_stripped = action.strip('\"')
            if action[0] != '"' or action_stripped not in self.entities_names:
                continue

            mapping[action_index] = self.entities_names[action_stripped]

        return mapping

    def get_query_without_table_hints(self):
        if not self.query:
            return ''
        toks = []
        for tok in self.query:
            if '@' in tok:
                parts = tok.split('@')
                if '.' in parts[0]:
                    toks.append(parts[0].split('.')[0] + '.' + parts[1])
                else:
                    toks.append(parts[1])
            else:
                toks.append(tok)
        return toks

    # def is_ambiguous_column(self, action_rhs: str, actions_sequence: List[str], action_index: int):
    #     """
    #     a column would be ambiguous if another table is used in the query and it has a column with the same name
    #     currently, return true only for join clauses
    #     """
    #     if actions_sequence[action_index-1].startswith('join_condition -> ') or \
    #             actions_sequence[action_index-2].startswith('join_condition -> '):
    #         return True
    #
    #     return False
    #     # column_table, column_name = action_rhs.strip('"').split('.')
    #     # tables_used = [a.split(' -> ')[1].strip('[]\"') for a in actions_sequence if a.startswith('table_name -> ')]
    #     # columns_with_same_name = set([a.split(' -> ')[1].strip('[]\"') for a in actions_sequence
    #     #                 if a.startswith('column_name -> ') and a.strip('[]\"').endswith(column_name)])
    #     # table_of_columns_with_same_name = set([c.split('.')[0] for c in columns_with_same_name])
    #     # if len(table_of_columns_with_same_name) > 1:
    #     #     return True
    #     # if column_table not in tables_used:
    #     #     return False
    #     # other_tables = [t for t in tables_used if t != column_table]
    #     # other_tables_columns = [[c.split(':')[-1] for c in self.knowledge_graph.neighbors[f'table:{t}']] for t in other_tables]
    #     # other_tables_columns_set = set([item for sublist in other_tables_columns for item in sublist])
    #     # return column_name in other_tables_columns_set
