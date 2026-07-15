import pandas as pd
from sklearn.metrics import accuracy_score,classification_report
def save_result(model_name,y_test,y_pred):
    report=classification_report(y_test,y_pred,output_dict=True)
    result={
        "MODEL":model_name,
        "ACCURACY":accuracy_score(y_test,y_pred),
        "F1_class_0":report["0"]["f1-score"],
        "F1_class_1":report["1"]["f1-score"],
        "Weighted_F1":report["weighted avg"]["f1-score"]
    }
    df=pd.DataFrame([result])
    from pathlib import Path

    csv_path = Path("report/model_comparison.csv")
    if csv_path.exists():
        df_old=pd.read_csv(csv_path)
        df_old = df_old[
        df_old["MODEL"] != model_name]

        df_final = pd.concat(
          [df_old, df],
          ignore_index=True
          )
    else:
        df_final=df
    print("Model Name =", model_name)
    print("CSV Path =", csv_path.resolve())

    print(df_final)    
    print("Saving CSV at:", csv_path.resolve())
    df_final.to_csv(csv_path,index=False)
    return result