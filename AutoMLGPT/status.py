from sklearn.model_selection import train_test_split

class Status():
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.init()
    
    def init(self):
        self.start_explore_prompt = ""
        self.how_to_explore_prompt = ""
        self.explore_data_thought = ""
        
        self.explore_data_result = "Nothing special."
        self.hyperopt_param_result = "Nothing special."
        
        self.explore = False
        self.preprocess = False
        self.model_select = False
        self.hyperopt = False
        
        self.explore_code = ""
        self.preprocess_code = ""
        self.model_select_code = ""
        self.hyperopt_code = ""
        
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.1, random_state=42)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = LGBMRegressor()
        self.best_param = {}
        
data = pd.read_csv("WineQT.csv")
X = data.drop('quality', axis=1)
y = data['quality']
status = Status(X, y)