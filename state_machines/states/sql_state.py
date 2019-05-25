import copy
import logging

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class SqlState:
    def __init__(self,
                 possible_actions,
                 enabled: bool=True):
        self.possible_actions = [a[0] for a in possible_actions]
        self.action_history = []
        self.tables_used = set()
        self.tables_used_by_columns = set()
        self.current_stack = []
        self.subqueries_stack = []
        self.enabled = enabled

    def take_action(self, production_rule: str) -> 'SqlState':
        if not self.enabled:
            return self

        new_sql_state = copy.deepcopy(self)

        lhs, rhs = production_rule.split(' -> ')
        rhs_tokens = rhs.strip('[]').split(', ')
        if lhs == 'table_name':
            new_sql_state.tables_used.add(rhs_tokens[0].strip('"'))
        elif lhs == 'column_name':
            new_sql_state.tables_used_by_columns.add(rhs_tokens[0].strip('"').split('@')[0])
        elif lhs == 'iue':
            new_sql_state.tables_used_by_columns = set()
            new_sql_state.tables_used = set()
        elif lhs == "source_subq":
            new_sql_state.subqueries_stack.append(copy.deepcopy(new_sql_state))
            new_sql_state.tables_used = set()
            new_sql_state.tables_used_by_columns = set()

        new_sql_state.action_history.append(production_rule)

        new_sql_state.current_stack.append([lhs, []])

        for token in rhs_tokens:
            is_terminal = token[0] == '"' and token[-1] == '"'
            if not is_terminal:
                new_sql_state.current_stack[-1][1].append(token)

        while len(new_sql_state.current_stack[-1][1]) == 0:
            finished_item = new_sql_state.current_stack[-1][0]
            del new_sql_state.current_stack[-1]
            if finished_item == 'statement':
                break
            if new_sql_state.current_stack[-1][1][0] == finished_item:
                new_sql_state.current_stack[-1][1] = new_sql_state.current_stack[-1][1][1:]

            if finished_item == 'source_subq':
                new_sql_state.tables_used = new_sql_state.subqueries_stack[-1].tables_used
                new_sql_state.tables_used_by_columns = new_sql_state.subqueries_stack[-1].tables_used_by_columns
                del new_sql_state.subqueries_stack[-1]

        return new_sql_state

    def get_valid_actions(self, valid_actions: dict):
        if not self.enabled:
            return valid_actions

        valid_actions_ids = []
        for key, items in valid_actions.items():
            valid_actions_ids += [(key, rule_id) for rule_id in valid_actions[key][2]]
        valid_actions_rules = [self.possible_actions[rule_id] for rule_type, rule_id in valid_actions_ids]

        actions_to_remove = {k: set() for k in valid_actions.keys()}

        current_clause = self._get_current_open_clause()

        if current_clause in ['where_clause', 'orderby_clause', 'join_condition', 'groupby_clause']:
            for rule_id, rule in zip(valid_actions_ids, valid_actions_rules):
                rule_type, rule_id = rule_id
                lhs, rhs = rule.split(' -> ')
                rhs_values = rhs.strip('[]').split(', ')
                if lhs == 'column_name':
                    rule_table = rhs_values[0].strip('"').split('@')[0]
                    if rule_table not in self.tables_used:
                        actions_to_remove[rule_type].add(rule_id)

                # if len(self.current_stack[-1][1]) < 2:
                #     # disable condition clause when same tables
                #     rule_table = rhs_values[0].strip('"').split('@')[0]
                #     last_table = self.action_history[-1].split(' -> ')[1].strip('[]"').split('@')[0]
                #     if rule_table == last_table:
                #         actions_to_remove[rule_type].add(rule_id)

        if current_clause in ['join_clause']:
            for rule_id, rule in zip(valid_actions_ids, valid_actions_rules):
                rule_type, rule_id = rule_id
                lhs, rhs = rule.split(' -> ')
                rhs_values = rhs.strip('[]').split(', ')
                if lhs == 'table_name':
                    candidate_table = rhs_values[0].strip('"')

                    if current_clause == 'join_clause' and len(self.current_stack[-1][1]) == 2:
                        if candidate_table in self.tables_used:
                            # trying to join an already joined table
                            actions_to_remove[rule_type].add(rule_id)

                    if 'join_clauses' not in self.current_stack[-2][1] and not self.current_stack[-2][0].startswith('join_clauses'):
                        # decided not to join any more tables
                        remaining_joins = self.tables_used_by_columns - self.tables_used
                        if len(remaining_joins) > 0 and candidate_table not in self.tables_used_by_columns:
                            # trying to select a single table but used other table(s) in columns
                            actions_to_remove[rule_type].add(rule_id)

        if current_clause in ['select_core']:
            for rule_id, rule in zip(valid_actions_ids, valid_actions_rules):
                rule_type, rule_id = rule_id
                lhs, rhs = rule.split(' -> ')
                rhs_values = rhs.strip('[]').split(', ')
                if self.current_stack[-1][1][0] == 'from_clause' or self.current_stack[-1][1][0] == 'join_clauses':
                    all_tables = set([a.split(' -> ')[1].strip('[]\"') for a in self.possible_actions if
                                      a.startswith('table_name ->')])
                    if len(self.tables_used_by_columns - self.tables_used) > 1:
                        # selected columns from more tables than selected, must join
                        if 'join_clauses' not in rhs:
                            actions_to_remove[rule_type].add(rule_id)
                    if len(all_tables - self.tables_used) <= 1:
                        # don't join 2 tables because otherwise there will be no more tables to join
                        # (assuming no joining twice and no sub-queries)
                        if 'join_clauses' in rhs:
                            actions_to_remove[rule_type].add(rule_id)
                if lhs == "table_name" and self.current_stack[-1][0] == "single_source":
                    candidate_table = rhs_values[0].strip('"')
                    if len(self.tables_used_by_columns) > 0 and candidate_table not in self.tables_used_by_columns:
                        # trying to select a single table but used other table(s) in columns
                        actions_to_remove[rule_type].add(rule_id)

                if lhs == 'single_source' and len(self.tables_used_by_columns) == 0 and rhs.strip('[]') == 'source_subq':
                    # prevent cases such as "select count ( * ) from ( select city.district from city ) where city.district = ' value '"
                    search_stack_pos = -1
                    while self.current_stack[search_stack_pos][0] != 'select_core':
                        # note - should look for other "gateaways" here (i.e. maybe this is not a dead end, if there is
                        # another source_subq. This is ignored here
                        search_stack_pos -= 1
                    if self.current_stack[search_stack_pos][1][-1] == 'where_clause':
                        # planning to add where/group/order later, but no columns were ever selected
                        actions_to_remove[rule_type].add(rule_id)

                    while self.current_stack[search_stack_pos][0] != 'query':
                        search_stack_pos -= 1
                    if 'orderby_clause' in self.current_stack[search_stack_pos][1]:
                        actions_to_remove[rule_type].add(rule_id)
                    if 'groupby_clause' in self.current_stack[search_stack_pos][1]:
                        actions_to_remove[rule_type].add(rule_id)

        new_valid_actions = {}
        new_global_actions = self._remove_actions(valid_actions, 'global',
                                                  actions_to_remove['global']) if 'global' in valid_actions else None
        new_linked_actions = self._remove_actions(valid_actions, 'linked',
                                                  actions_to_remove['linked']) if 'linked' in valid_actions else None

        if new_linked_actions is not None:
            new_valid_actions['linked'] = new_linked_actions
        if new_global_actions is not None:
            new_valid_actions['global'] = new_global_actions

        # if len(new_valid_actions) == 0 and valid_actions:
        #     # should not get here! implies that a rule should have been disabled in past (bug in this parser)
        #     # log and do not remove rules (otherwise crashes)
        #     # logger.warning("No valid action remains, error in sql decoding parser!")
        #     # logger.warning("Action history: " + str(self.action_history))
        #     # logger.warning("Tables in db: " + ', '.join([a.split(' -> ')[1].strip('[]\"') for a in self.possible_actions if a.startswith('table_name ->')]))
        #
        #     return valid_actions

        return new_valid_actions

    @staticmethod
    def _remove_actions(valid_actions, key, ids_to_remove):
        if len(ids_to_remove) == 0:
            return valid_actions[key]

        if len(ids_to_remove) == len(valid_actions[key][2]):
            return None

        current_ids = valid_actions[key][2]
        keep_ids = []
        keep_ids_loc = []

        for loc, rule_id in enumerate(current_ids):
            if rule_id not in ids_to_remove:
                keep_ids.append(rule_id)
                keep_ids_loc.append(loc)

        items = list(valid_actions[key])
        items[0] = items[0][keep_ids_loc]
        items[1] = items[1][keep_ids_loc]
        items[2] = keep_ids

        if len(items) >= 4:
            items[3] = items[3][keep_ids_loc]
        return tuple(items)

    def _get_current_open_clause(self):
        relevant_clauses = [
            'where_clause',
            'orderby_clause',
            'join_clause',
            'join_condition',
            'select_core',
            'groupby_clause',
            'source_subq'
        ]
        for rule in self.current_stack[::-1]:
            if rule[0] in relevant_clauses:
                return rule[0]

        return None
