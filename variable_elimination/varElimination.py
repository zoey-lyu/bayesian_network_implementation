import pandas as pd
import numpy as np
import itertools
pd.options.mode.chained_assignment = None  # default='warn'


class BayesModel():
    def __init__(self, edgelst):
        self.edges = []
        self.nodes = list(set([k for i in edgelst for k in i]))
        self.edges.extend(e for e in edgelst)
        self.cpd_lst = []
        self.parents = {}
        self.childs = {}
        for n in self.nodes:
            self.childs[n] = []
            for tup in self.edges:
                if n == tup[0]:
                    self.childs[n].append(tup[1])

    def add_cpds(self, *cpds): #added cpd objects, not df
        for cpd in cpds:
            self.cpd_lst.append(cpd)
            self.parents[cpd.variable] = cpd.evidence


class TabularCPD():
    def __init__(self, variable, values, evidence = None):
        self.variable = variable
        self.values = values
        self.evidence = evidence
        self.cpd = self.build_table()

    def build_table(self):
        if self.evidence:
            t_table = list(itertools.product([0, 1], repeat= (len(self.evidence)+1)))
            col_name = [i for i in self.evidence]
            col_name.append(self.variable)
            table = pd.DataFrame(t_table, columns = col_name)
            value_lst = list(np.asarray(self.values).flatten('F'))
            table['p'] = value_lst
        else:
            table = pd.DataFrame({self.variable:[0,1], 'p': self.values[0]})

        return table

    @staticmethod
    def normalize(df):
        norm_col = df['p'].sum(axis = 0)
        df['p'] = df['p']/norm_col
        return df

    @staticmethod
    def multiply(df1, df2):
        '''if given cpd, need to use cpd.cpd as input 
        so that it plugs in a dataframe'''
        merge_on = list(df1.columns.intersection(df2.columns))
        merge_on.remove('p')
        merge = pd.merge(df1, df2, on = merge_on)
        merge['p'] = merge.p_x * merge.p_y
        mul_table = merge.drop(['p_x','p_y'], axis = 1)
        return mul_table

    @staticmethod
    def marginalize(df, var_lst):
        s1 = set(df.columns)
        s1.remove('p')
        gb_input = list(s1 - set(var_lst))
        marg = df.groupby(gb_input).sum()
        marg = marg.reset_index()
        return marg


class VarElimination():
    def __init__(self, bayes_model):
        self.model = bayes_model
    
    @staticmethod
    def plug_in_evidence(result_table, evi_dict):
        assigned_var = list(evi_dict.keys())
        for v in assigned_var:
            if v in list(result_table.columns):
                result_table = result_table.loc[result_table[v] == evi_dict[v]]
                result_table = TabularCPD.normalize(result_table)
        return result_table
    
    @staticmethod
    def get_related_del(var,df):
        to_mul = []
        to_del = []
        for i in range(len(df)):
            sin_cpd = df[i] # df
            if var in list(sin_cpd.columns):
                to_mul.append(sin_cpd) # add df
                to_del.append(i)
        for j in range(len(to_del)):
            df.pop(to_del[j]) # update the list of df
            # off set to delete list
            for k in range(len(to_del)):
                to_del[k] -= 1
        return to_mul

    def query(self, var_lst, evi_dict = None):

        '''eliminate order is based on min relatives'''
        all_nodes = self.model.nodes
        to_eliminate = set(all_nodes) - set(var_lst) - set(evi_dict.keys() if evi_dict else [])
        def get_num_relatives(var):
            return ((len(self.model.parents[var]) if self.model.parents[var] else 0) + (len(self.model.childs[var]) if self.model.childs[var] is not None else 0)) 
        order = sorted(to_eliminate, key = get_num_relatives)
        print('suggested order',order)
        
        all_tables = [x.cpd for x in self.model.cpd_lst] # all dfs

        # canceling out the evidence probability tables first
        if evi_dict:
            for var in list(evi_dict.keys()):
                to_del = []
                for i in range(len(all_tables)):
                    sin_cpd = all_tables[i]
                    if var in list(sin_cpd.columns) and len(sin_cpd.columns) == 2:
                        to_del.append(i)
                for j in range(len(to_del)):
                    all_tables.pop(to_del[j]) # update the list of df
                    for k in range(len(to_del)):
                        to_del[k] -= 1

        for var in order:

            related = self.get_related_del(var,all_tables)
            result = related[0] #take the first df
            for i in range(len(related)-1):
                result = TabularCPD.multiply(result,related[i+1])
            result = TabularCPD.marginalize(result,[var])
            result = result.drop(var, axis = 1)
            all_tables.insert(0,result) 
        
        if len(all_tables) == 1:
            result = TabularCPD.normalize(result)
            if evi_dict:
                result = self.plug_in_evidence(result,evi_dict)
            return result

        else:
            var_left = set()
            for df in all_tables:
                var_left.update(list(df.columns))
            var_left.remove('p')

            for var in var_left:
                related = self.get_related_del(var,all_tables)      
                result = related[0]

                for i in range(len(related)-1):
                    result = TabularCPD.multiply(result,related[i+1])
                all_tables.insert(0,result)

            result = TabularCPD.normalize(result)
            if evi_dict:
                result = self.plug_in_evidence(result,evi_dict)

            return result