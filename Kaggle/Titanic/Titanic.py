import pandas as pd
from pycaret.classification import *

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

setup(data=train, target='Survived', ignore_features=['PassengerId', 'Name'], session_id=1, silent=True)

best_model = compare_models()
finalized_model = finalize_model(best_model)
prediction = predict_model(finalized_model, data=test)

submission = pd.DataFrame({
        'PassengerId': test.PassengerId,
        'Survived': prediction.Label
    })
submission.to_csv('submission.csv', index=False)