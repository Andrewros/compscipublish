from flask import Flask, render_template, request, session, redirect, flash
from flask_session import Session
import torch
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import sqlite3
from PIL import Image
from werkzeug.security import check_password_hash, generate_password_hash
import os
from functools import wraps
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get("user_id") is None:
            return redirect("/login")
        return f(*args, **kwargs)
    return decorated_function
training=True
app=Flask(__name__)
CLASSES = ['Andrew', 'Brad', 'Matthew']
preprocessedsteps=transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize([32, 32]),
    transforms.ToTensor()
])
app.config["SESSION_TYPE"] = "filesystem"
Session(app)
USERNAME=0
PASSWORD=1
ID=2
DATASET_PATH="static/datasets"
class ImageModel(nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self.layers=nn.Sequential(
            nn.Conv2d(1,6,3), #6x30x30
            nn.ReLU(),
            nn.Conv2d(6,16,3), #16x28x28
            nn.ReLU(),
            nn.MaxPool2d(2,2),#16x14x14
            nn.Flatten(),
            nn.Linear(16*14*14, 32),
            nn.ReLU(),
            nn.Linear(32, outputs)
        )
    def forward(self, x):
        return self.layers(x)

@app.route("/login", methods=["GET", "POST"])
def login():
    # Forget any user_id
    session.clear() 
    # User reached route via POST (as by submitting a form via POST)
    if request.method == "POST":    
      # Ensure username was submitted
        if not request.form.get("username"):
            return "must provide username"    
      # Ensure password was submitted
        elif not request.form.get("password"):
            return "must provide password"   
      # Query database for username
        db=sqlite3.connect("static/users.db")
        cursor=db.cursor()
        try:
            rows = cursor.execute("SELECT * FROM users WHERE username = ?",
                            request.form.get("username")).fetchone()
            # Ensure username exists and password is correct
            if not check_password_hash(rows[PASSWORD],
                                                        request.form.get("password")):
                return render_template("error.html", message="invalid username and/or password")

            # Remember which user has logged in
            session["user_id"] = rows[ID]

            # Redirect user to home page
            return redirect("/")
        except:
            return render_template("error.html", message="Something you entered was invalid")

  # User reached route via GET (as by clicking a link or via redirect)
    else:
        return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    session.clear()
    if request.method == "POST":
        # Ensure username was submitted
    
        if not request.form.get("username"):
          return "must provide username"
    
        # Ensure password was submitted
        elif not request.form.get("password") or not request.form.get(
            "password2") or request.form.get("password") != request.form.get(
              "password2"):
          return "must provide password"
        hashedpassword = generate_password_hash(request.form.get("password"))
        db=sqlite3.connect("static/users.db")
        cursor=db.cursor()
        cursor.execute("INSERT INTO users(username, password) VALUES (?, ?)",
                   (request.form.get("username"), hashedpassword))
        db.commit()
        rows = cursor.execute("SELECT * FROM users WHERE username = ?",
                          (request.form.get("username"), )).fetchone()
        print(rows)
        session["user_id"] = rows[ID]
    # Remember which user has logged in

    # Redirect user to home page
        return redirect("/")

  # User reached route via GET (as by clicking a link or via redirect)
    else:
        return render_template("register.html")


@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    if request.method=="GET":
        return render_template("index.html")
    else:
        # try:
        file=request.files["image"].stream
        file=Image.open(file)
        transformed=preprocessedsteps(file)
        db=sqlite3.connect("static/users.db")
        cursor=db.cursor()
        data=cursor.execute("SELECT class FROM classes WHERE id=?", (session["user_id"],)).fetchall()
        classes=[datum[0].capitalize() for datum in data]
        classes.extend(CLASSES)
        classes.sort()
        model=ImageModel(len(classes))
        if os.path.exists(f"{DATASET_PATH}/train{session['user_id']}/data.pt"):
            model.load_state_dict(torch.load(f"{DATASET_PATH}/train{session['user_id']}/data.pt"))
        else:
            model.load_state_dict(torch.load("static/model.pt"))
        model.eval()
        with torch.inference_mode():
            yhat=torch.argmax(model(transformed.unsqueeze(1)))
        
        return render_template("prediction.html", name=classes[yhat.numpy()])
        # except:
        #     return redirect("/")
@app.route("/train", methods=["GET", "POST"])
@login_required
def train():
    if request.method=="GET":
        return render_template("train.html")
    else:
        classname=request.form.get("class").capitalize()
        if not os.path.exists(f"{DATASET_PATH}/train{session['user_id']}"):
            os.system(f"mkdir {DATASET_PATH}/train{session['user_id']}")
            for clas in CLASSES:
                os.system(f"cp -R static/{clas} {DATASET_PATH}/train{session['user_id']}")
        
        if os.path.exists(f"{DATASET_PATH}/train{session['user_id']}/{classname}"):
            return render_template("error.html", message="Class is already being used")
        
        images=request.files.getlist("images")
        if len(images)!=12:
            return render_template("error.html", message="Must provide exactly 12 images")
        #WRITE LOTS OF CODE BEFORE MAKING THE DIRECTORY
        os.system(f"mkdir {DATASET_PATH}/train{session['user_id']}/{classname}")
        for image in images:
            filename = image.filename
            filename=filename.replace("/", "")
            save_path = f"{DATASET_PATH}/train{session['user_id']}/{classname}/{filename}"
            image.save(save_path)
        traindataset=torchvision.datasets.ImageFolder(root=f"static/datasets/train{session['user_id']}", transform=preprocessedsteps)
        trainloader=DataLoader(traindataset, 2, shuffle=True)
        lossfn=nn.CrossEntropyLoss()
        db=sqlite3.connect("static/users.db")
        cursor=db.cursor()
        data=cursor.execute("SELECT class FROM classes WHERE id=?", (session["user_id"],)).fetchall()
        classes=[datum[0].capitalize() for datum in data]
        classes.extend(CLASSES)
        while True:
            model=ImageModel(len(classes)+1)
            optimizer=torch.optim.SGD(params=model.parameters(), lr=0.008, momentum=0.8)
            epochs=16
            losses=[]
            for epoch in range(epochs):
                for x,y in trainloader:
                    model.train()
                    yhat=model(x)
                    loss=lossfn(yhat, y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if epoch==epochs-1 and loss.item()<0.5:
                        losses.append(loss.item())
                print(f"Epoch: {epoch} loss:{loss.item()}")

            if len(losses)>4 and loss.item()<0.8:
                break
        torch.save(model.state_dict(), f"{DATASET_PATH}/train{session['user_id']}/data.pt")
        db=sqlite3.connect("static/users.db")
        cursor=db.cursor()
        cursor.execute("INSERT INTO classes (id, class) VALUES (?,?)", (session["user_id"], classname,))
        db.commit()
        flash("Training complete")
        return redirect("/")
@app.route("/reset", methods=["GET", "POST"])
@login_required
def reset():
    if request.method=="GET":
        return render_template("reset.html")
    else:
        db=sqlite3.connect("static/users.db")
        cursor=db.cursor()
        cursor.execute("DELETE from classes WHERE id=?", (session['user_id'],))
        db.commit()
        os.system(f"rm -rf static/datasets/train{session['user_id']}")
        os.system(f"mkdir {DATASET_PATH}/train{session['user_id']}")
        for clas in CLASSES:
            os.system(f"cp -R static/{clas} {DATASET_PATH}/train{session['user_id']}")
        return redirect("/")
        


@app.route("/logout")
def logout():
  """Log user out"""

  # Forget any user_id
  session.clear()

  # Redirect user to login form
  return redirect("/")

if __name__=="__main__":
    app.run()