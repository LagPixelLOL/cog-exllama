from cog import BasePredictor, Input, Path

class Predictor(BasePredictor):
    
    def setup(self) -> None:
        return

    def predict(
        self,
    ) -> None:
        
        import os
        os.system('ls .git -lha')
        
        print("Hello, world!")
