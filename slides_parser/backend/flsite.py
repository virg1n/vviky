import sqlite3
import os
from flask import Flask, render_template, request, g, flash, abort, redirect, url_for
from FDataBase import FDataBase
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from UserLogin import UserLogin
import sys
import time
import io
from pptx import Presentation
from threading import Thread
import pathlib
import shutil
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# from slides_parser.last import generate_presentation
from last import generate_presentation


DATABASE = '/tmp/flsite.db'
DEBUG = True
SECRET_KEY = 'SECRET_KEY'

# def create_app():
app = Flask(__name__)
app.config.from_object(__name__)
app.config.update(dict(DATABASE=os.path.join(app.root_path,'flsite.db')))

login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message = "Авторизуйтесь для доступа к закрытым страницам"
login_manager.login_message_category = "success"

delete_list = []
path_to_slides_parser = pathlib.Path().resolve()

LIMIT_RECENT_PRESENTATIONS = 4
SECONDS_FOR_DELETE_PPTX_FROM_DIRECTORY = 180


@login_manager.user_loader
def load_user(user_id):
    print("load_user")
    return UserLogin().fromDB(user_id, dbase)


def connect_db():
    conn = sqlite3.connect(app.config['DATABASE'], check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def create_db():
    """Вспомогательная функция для создания таблиц БД"""
    db = connect_db()
    with app.open_resource('sq_db.sql', mode='r') as f:
        db.cursor().executescript(f.read())
    db.commit()
    db.close()

def get_db():
    '''Соединение с БД, если оно еще не установлено'''
    if not hasattr(g, 'link_db'):
        g.link_db = connect_db()
    return g.link_db


dbase = None
@app.before_request
def before_request():
    """Установление соединения с БД перед выполнением запроса"""
    global dbase
    db = get_db()
    dbase = FDataBase(db)


# @app.teardown_appcontext
# def close_db(error):
#     '''Закрываем соединение с БД, если оно было установлено'''
#     if hasattr(g, 'link_db'):
#         g.link_db.close()


@app.route("/")
def index():
    if current_user.is_authenticated:
        return render_template("index_for_registered.html")
    return render_template("index.html")


@app.route("/login", methods=["POST", "GET"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('profile'))

    if request.method == "POST":
        user = dbase.getUserByEmail(request.form['email'])
        if user and check_password_hash(user['psw'], request.form['password']):
            userlogin = UserLogin().create(user)
            rm = True if request.form.get('remainme') else False
            login_user(userlogin, remember=rm)
            return redirect(request.args.get("next") or url_for("profile", NAME="asdsdf"))

        flash("email or password is incorrect", "error")

    return render_template("login.html")


@app.route("/register", methods=["POST", "GET"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('profile'))
    if request.method == "POST":
        if len(request.form['name']) > 4 and len(request.form['email']) > 4 \
            and len(request.form['password']) > 4:
            hash = generate_password_hash(request.form['password'])
            res = dbase.addUser(request.form['name'], request.form['email'], hash)
            if res:
                user = dbase.getUserByEmail(request.form['email'])
                userlogin = UserLogin().create(user)
                # rm = True if request.form.get('remainme') else False
                login_user(userlogin, remember=False)
                return redirect(request.args.get("next") or url_for("profile"))
                
            else:
                flash("This email has already been used", "error")
        else:
            flash("The fields are filled in incorrectly", "error")

    return render_template("register.html")


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("Вы вышли из аккаунта", "success")
    return redirect(url_for('login'))
    

@app.route('/create', methods=["POST", "GET"])
@login_required
def create():
    global delete_list
    id = current_user.get_id()
    name = dbase.getUserByID(id)['name']
    if request.method == "POST":
        timecode = time.time()
        try:
            if random.randint(1, 100) == 1:
                delete_folder()
            else: 
                delete_by_id()
            topic = request.json.get('topic')
            username = request.json.get('username')
            slides = request.json.get('slides')
            color_theme = request.json.get('color_theme')
            
            # NEED TO CHANGE
            filename = str(username+str(timecode))
            presentation_data = ""
            generate_presentation(topic, username, int(slides), filename, timecode=str(timecode), color_of_theme=color_theme)
            with open(fr'{path_to_slides_parser}\static\presentations\{filename}.pptx', 'rb') as file:
                presentation_data = file.read()
            dbase.addPresentation(topic, int(id), str(f"../static/images/const/{username+str(timecode)}.jpeg"), f'../static/presentations/{filename}.pptx', presentation_data)

            # dbase.updatePresentationBlob(f'../static/presentations/{filename}.pptx', presentation_data)
            presentations = dbase.getPresentationByCreator(id, LIMIT_RECENT_PRESENTATIONS*2)
            try:
                if len(presentations) > LIMIT_RECENT_PRESENTATIONS:
                    path_to_presentation = fr"{path_to_slides_parser}{presentations[-1]['download_path'][2:]}"
                    try:
                        os.remove(fr"{path_to_slides_parser}{presentations[-1]['picture'][2:]}")
                    except:
                        pass
                    delete_list.append({
                    "path_to_presentation":path_to_presentation,
                    "time":time.time(),
                    "user_id":id
                })
                    dbase.deletePresentations(len(presentations) - LIMIT_RECENT_PRESENTATIONS, id)
                    presentations.reverse()
                    for i in presentations:
                        os.remove(fr"{path_to_slides_parser}{presentations[i]['download_path'][2:]}")
                        print("DELETED with sql")
            except Exception as e:
                print(e)

            return ({"preview":str(f"../static/images/const/{username+str(timecode)}.jpeg"), 
                    "path_to_download":str(f"../static/presentations/{filename}.pptx")})
        except Exception as e:
            try:
                username = request.json.get('username')
                filename = str(username+str(timecode))
                os.remove(fr'{path_to_slides_parser}/{filename}.pptx')
            except:
                pass
            print(e)
            return ({"error":"Sorry, an error has occurred. Try again or change the prompt"})

    return render_template("create.html", NAME=name)

@app.route('/profile')
@login_required
def profile():
    global delete_list
    id = current_user.get_id()

    presentations_to_send = []
    try:
        presentations = dbase.getPresentationByCreator(id, limit=LIMIT_RECENT_PRESENTATIONS)
        presentations.reverse()
        delete_by_id()
        for i in range(len(presentations)):
            try:
                presentation_blob = presentations[i]['presentation_blob']

                presentation_data = io.BytesIO(presentation_blob)
                pptxx = Presentation(presentation_data)
            
            # NEED TO CHANGE
                path_to_presentation = fr"{path_to_slides_parser}{presentations[i]['download_path'][2:]}"
                pptxx.save(path_to_presentation)
                presentations_to_send.append({'pic':presentations[i]['picture'],
                                            'tit':presentations[i]['title'],
                                            'download_path':presentations[i]['download_path']})
                delete_list.append({
                    "path_to_presentation":path_to_presentation,
                    "time":time.time(),
                    "user_id":id

                })
            except:
                pass
    
            
    except Exception as e:
        print(e)
        print("ERROR ON PROFILE")
    name = dbase.getUserByID(id)['name']
    return render_template("profile.html", NAME=name, pictures=presentations_to_send[::-1]) #, pictures=['picture'], titles=['title']

def delete_by_id():
    global delete_list
    try:
        for i in delete_list:
            if SECONDS_FOR_DELETE_PPTX_FROM_DIRECTORY < (time.time() - i['time']):
                presentations = dbase.getPresentationByCreator(i['user_id'], limit=LIMIT_RECENT_PRESENTATIONS)
                for j in range(len(presentations)):
                    try:
                        os.remove(fr"{path_to_slides_parser}{presentations[j]['download_path'][2:]}")
                    except:
                        pass
    except:
        pass


def delete_folder():
    try:
        dirpath = fr"{path_to_slides_parser}/static/presentations"
        for filename in os.listdir(dirpath):
            filepath = os.path.join(dirpath, filename)
            try:
                shutil.rmtree(filepath)
            except OSError:
                os.remove(filepath)
    except:
        pass

if __name__ == "__main__":
    try:
        create_db()
    finally:
        app.run(host='0.0.0.0', debug=True)

