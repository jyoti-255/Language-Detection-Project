#import lib
from flask import *
import pandas as pd
from nltk import word_tokenize
from string import punctuation
from pickle import load

#text cleaning
def clean_data(txt):
    txt = txt.lower()
    txt = txt.replace('"', "")  
    txt = word_tokenize(txt)
    txt = [t for t in txt if t not in punctuation]  
    txt = "".join(txt) 
    return txt

# restore the model and vector
f=open("lang_detection_model.pkl","rb")
model=load(f)
f.close()
f=open("lang_detection_vector.pkl","rb")
tv=load(f)
f.close()

app=Flask(__name__)
@app.route("/",methods=["GET","POST"])
def home():
    if request.method=="POST":
        txt=request.form["txt"]
        ctxt=clean_data(txt)
        vtxt=tv.transform([txt])
        res=model.predict(vtxt)
        return render_template("home.html",msg=res[0])
    else:
        return render_template("home.html")

if __name__=="__main__":
    app.run(debug=True,use_reloader=True)

          