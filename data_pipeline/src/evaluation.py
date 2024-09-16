import mlflow
from utils import logger
from sklearn.metrics import f1_score


def evaluation(model, test_data):
    try:
        actual_data = (test_data['anomaly'] > 0).astype(int)
        pred = model.predict(test_data.values)
        if -1 in pred:
            pred = [1 if p == -1 else 0 for p in pred]
        
        F1 = f1_score(actual_data, pred, average='macro')
        mlflow.log_metric("f1_score", F1)

        return F1
    except Exception as e:
        logger.error(e)
        raise e
    

def main():
    try:
        
        return
    except Exception as e:
        logger.error(e)
        raise e
    

if __name__ == "__main__":
    main()