import yaml
import pymysql.cursors
import warnings
import pymysql

class DB:
    def __init__(self):
        config = {}
        with open("./config/config.yml","r") as f:
            config = yaml.load(f)
        self.config = config['db']

        self.connection = pymysql.connect(host = config["db"]["host"],
                                     user = config["db"]["user"],
                                     password = config["db"]["password"],
                                     db = config["db"]["database"],
                                     charset = config["db"]["charset"],
                                     cursorclass = pymysql.cursors.DictCursor)

    def getTweets(self, offset, limit, table_name = 'emoji_tweets3'):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.connection.cursor() as cursor:
                sql = 'SELECT tweet FROM ' + table_name + ' LIMIT ' + str(limit) + " OFFSET " + str(offset) + ";"
                cursor.execute(sql)
                return cursor.fetchall()

    def __del__(self):
        self.connection.close()



