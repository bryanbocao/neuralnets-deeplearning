from scipy.stats import spearmanr

def get_distance_prob_correlation(data):
    x = []
    x.extend(data.alpha_pairs['same'][3])
    x.extend(data.alpha_pairs['diff'][3])
    x.extend(data.beta_pairs['same'][3])
    x.extend(data.beta_pairs['diff'][3])
    #print ("x,", x)
    
    y = []
    y.extend(data.alpha_pairs['same'][5])
    y.extend(data.alpha_pairs['diff'][5])
    y.extend(data.beta_pairs['same'][5])
    y.extend(data.beta_pairs['diff'][5])
    #print("y, ", y)
    
    return spearmanr(x, y, axis=None)

def get_cosine_prob_correlation(data):
    x = []
    x.extend(data.alpha_pairs['same'][2])
    x.extend(data.alpha_pairs['diff'][2])
    x.extend(data.beta_pairs['same'][2])
    x.extend(data.beta_pairs['diff'][2])
    #print ("x,", x)
    
    y = []
    y.extend(data.alpha_pairs['same'][5])
    y.extend(data.alpha_pairs['diff'][5])
    y.extend(data.beta_pairs['same'][5])
    y.extend(data.beta_pairs['diff'][5])
    #print("y, ", y)
    
    return spearmanr(x, y, axis=None)

def get_distance_acc_correlation(data):
    x = []
    x.extend(data.alpha_pairs['same'][3])
    x.extend(data.alpha_pairs['diff'][3])
    x.extend(data.beta_pairs['same'][3])
    x.extend(data.beta_pairs['diff'][3])
    #print ("x,", x)
    
    y = []
    y.extend(data.alpha_pairs['same'][4])
    y.extend(data.alpha_pairs['diff'][4])
    y.extend(data.beta_pairs['same'][4])
    y.extend(data.beta_pairs['diff'][4])
    #print("y, ", y)
    
    return spearmanr(x, y, axis=None)

def get_cosine_acc_correlation(data):
    x = []
    x.extend(data.alpha_pairs['same'][2])
    x.extend(data.alpha_pairs['diff'][2])
    x.extend(data.beta_pairs['same'][2])
    x.extend(data.beta_pairs['diff'][2])
    #print ("x,", x)
    
    y = []
    y.extend(data.alpha_pairs['same'][4])
    y.extend(data.alpha_pairs['diff'][4])
    y.extend(data.beta_pairs['same'][4])
    y.extend(data.beta_pairs['diff'][4])
    #print("y, ", y)
    
    return spearmanr(x, y, axis=None)
    