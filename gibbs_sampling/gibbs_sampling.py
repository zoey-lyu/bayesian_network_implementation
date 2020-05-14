from varElimination import *
import matplotlib.pyplot as plt

class GibbsSampling():
    def __init__(self, bayes_model):
        self.model =  bayes_model

    def query(self, var, evi_dict = None, step = 1000, burn_in = None):
        # takes in var string
        self.variable = var
        self.evi_var = list(evi_dict.keys()) if evi_dict else []
        self.nodes_to_sample = list(set(self.model.nodes) - set(self.evi_var))
        # sampling order will be based on this nodes_to_sample list

        #sample dictionary: {'V1':[list of all the sample 'V1' values], 'V2' ...} 
        sample = {}
        prob_var = {}
        # initialization
        for node in self.nodes_to_sample:
            prob_var[node] = []
            var_init = np.random.randint(low = 0, high = 2)
            sample[node] = [var_init]  
        
        all_tables = [i.cpd for i in self.model.cpd_lst]

        burn = burn_in if burn_in else 0
        iter_num = step + burn

        for step in range(iter_num):
            # find tables related to the variable being sampled
            for node in self.nodes_to_sample:

                related = []
                for sin_cpd in all_tables:
                    if node in list(sin_cpd.columns):
                        related.append(sin_cpd)
                result = related[0]
                # calculate numerator
                for i in range(len(related)-1):
                    result = TabularCPD.multiply(result,related[i+1])
                # denominator is just a normalizer

                # if evi_dict is given, plug in e 
                if evi_dict:
                    for evi_var in self.evi_var:
                        if evi_var in list(result.columns):
                            result = result.loc[result[evi_var] == evi_dict[evi_var]]
                            result = TabularCPD.normalize(result)
            
                # plug in other sample values(with the latest update):
                for sam_var in list(set(self.nodes_to_sample) - set([node])):
                    if  sam_var in list(result.columns):
                        result = result.loc[result[sam_var] == sample[sam_var][0]]
                        result = TabularCPD.normalize(result)
                
                if result.p.isna().values.any():
                    # when there's nan need to readjusted
                    # strategy: VE or randomize again
                    #samp_prob = VarElimination.query([node],evi_dict).p.to_list()
                    #samp = np.random.choice([0,1],p = samp_prob)
                    samp = np.random.choice([0,1])
                    sample[node].insert(0,samp)
                    if step > (burn - 1):
                        prob_var[node].append(1-sum(sample[node][:- burn - 1])/(step - burn +1))

                else:
                    samp = np.random.choice([0,1],p = result.p.to_list())
                    sample[node].insert(0,samp)
                    if step > (burn - 1):
                        prob_var[node].append(1-sum(sample[node][:-burn - 1])/(step - burn +1))    
        return(prob_var[var])

