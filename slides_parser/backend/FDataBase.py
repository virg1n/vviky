import sqlite3
import time
import math
import re
from flask import url_for

class FDataBase:
    def __init__(self, db):
        self.__db = db
        self.__cur = db.cursor()

    def getMenu(self):
        sql = '''SELECT * FROM mainmenu'''
        try:
            self.__cur.execute(sql)
            res = self.__cur.fetchall()
            if res: return res
        except:
            print("Ошибка чтения из БД")
        return []

    def addPresentation(self, title, creator, picture, download_path, presentation_blob):
        try:

            tm = math.floor(time.time())
            self.__cur.execute("INSERT INTO presentations VALUES(NULL, ?, ?, ?, ?, ?, ?)", (title, creator, picture, tm, download_path, presentation_blob))
            self.__db.commit()
        except sqlite3.Error as e:
            print("Ошибка добавления статьи в БД "+str(e))
            return False

        return True
    
    def updatePresentationBlob(self, presentation_path, new_presentation_blob):
        try:
            self.__cur.execute(
                "UPDATE presentations SET presentation_blob = ? WHERE download_path = ?",
                (new_presentation_blob, presentation_path)
            )
            self.__db.commit()
        except sqlite3.Error as e:
            print("Error updating presentation blob: " + str(e))
            return False
        

    def deletePresentations(self, number_of_presentations, creator):
        try:

            self.__cur.execute("""
    DELETE FROM presentations
    WHERE creator = ? AND time = (
        SELECT time FROM presentations WHERE creator = ? ORDER BY time LIMIT ?
    );
""", (creator, number_of_presentations, creator))
            self.__db.commit()
        except sqlite3.Error as e:
            print("Error to delete from presentations table "+str(e))
            return False

        return True

    def getPost(self, alias):
        try:
            self.__cur.execute(f"SELECT title, text FROM posts WHERE url LIKE '{alias}' LIMIT 1")
            res = self.__cur.fetchone()
            if res:
                return res
        except sqlite3.Error as e:
            print("Ошибка получения статьи из БД "+str(e))

        return (False, False)

    def getUserByEmail(self, email):
        try:
            self.__cur.execute(f"SELECT * FROM users WHERE email = '{email}' LIMIT 1")
            res = self.__cur.fetchone()
            if not res:
                print("Пользователь не найден")
                return False

            return res
        except sqlite3.Error as e:
            print("Ошибка получения данных из БД "+str(e))

        return False

    def addUser(self, name, email, hpsw):
        try:
            self.__cur.execute(f"SELECT COUNT() as `count` FROM users WHERE email LIKE '{email}'")
            res = self.__cur.fetchone()
            if res['count'] > 0:
                print("Пользователь с таким email уже существует")
                return False

            tm = math.floor(time.time())
            self.__cur.execute("INSERT INTO users VALUES(NULL, ?, ?, ?, ?)", (name, email, hpsw, tm))
            self.__db.commit()
        except sqlite3.Error as e:
            print("Ошибка добавления пользователя в БД "+str(e))
            return False

        return True
    

    def getUser(self, user_id):
        try:
            self.__cur.execute(f"SELECT * FROM users WHERE id = {user_id} LIMIT 1")
            res = self.__cur.fetchone()
            if not res:
                print("Пользователь не найден")
                return False

            return res
        except sqlite3.Error as e:
            print("Ошибка получения данных из БД "+str(e))

        return False

    def getPresentationByCreator(self, id, limit):
        try:
            self.__cur.execute(f"SELECT title, picture, download_path, presentation_blob FROM presentations WHERE creator = {id} ORDER BY time DESC LIMIT {limit}")
            res = self.__cur.fetchall()
            if not res:
                print(f"Presentation for id {id} were not found")
                return False

            return res
        except sqlite3.Error as e:
            print("Ошибка получения данных из БД "+str(e))

        return False

    def getUserByID(self, id):
        try:
            self.__cur.execute(f"SELECT * FROM users WHERE id = '{id}' LIMIT 1")
            res = self.__cur.fetchone()
            if not res:
                print("Пользователь не найден")
                return False

            return res
        except sqlite3.Error as e:
            print("Ошибка получения данных из БД "+str(e))

        return False

