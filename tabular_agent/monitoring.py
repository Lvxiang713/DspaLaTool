"""Presentation-only sample normalization, preserving recorded loss sums."""
def display_history(rows, training_n=None):
    result=[]
    for row in rows:
        row=dict(row)
        n=row.get("training_n",training_n)
        if n and n>0 and "total_loss_sum" in row:
            row["total_loss_mean"]=row["total_loss_sum"]/n
        result.append(row)
    return result
