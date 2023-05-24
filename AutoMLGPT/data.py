class DataLoader():
    
    def __init__(self, chain):
        self.datasets = {}
        self.chain = chain
        
    def select(self, name: str):
        if name not in self.datasets:
            pass
        else:
            return self.datasets[name]
        
    def show(self):
        print(self.datasets.keys())
    
    def load(self):
        
        for dirn in os.listdir('../../kaggle'):
            
            if not os.path.isdir(dirn):
                continue

            try:
            
                train_path = os.path.join('kaggle', dirn, 'train.csv')
                train_df = pd.read_csv(train_path)

                test_path = os.path.join('kaggle', dirn, 'test.csv')
                test_df = pd.read_csv(test_path)

                label = list(set(train_df.columns) - set(test_df.columns))[0]

                self.datasets[dirn] = {
                    'train': train_df,
                    'test': test_df,
                    'label': label,
                }
            except Exception as e:
                print(e)
                
                
data_loader = DataLoader(chain)
data_loader.load()
