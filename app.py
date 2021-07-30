from logging import debug
from flask import Flask,render_template,request
import cancer as c

app=Flask(__name__)
@app.route("/",methods=['GET','POST'])
def hello(cp=0):
    if request.method=='POST':
        features=[int(x) for x in request.form.values()]
        cancer_pred=c.cancer_prediction(features)
        cp=cancer_pred
        if cp==2:
            cp='Benign'
        else:
            cp='Melignant'

    return render_template("index.html",machine_pred=cp)

if __name__=="__main__":
    app.run(debug=True)