# -*- encoding: utf-8 -*-
import MySQLdb as mysqldb


def toMessage(tableName):
    # 连接数据库
    dbfile = 'datamining'
    con = mysqldb.connect(host='localhost', port=3306, user='root', passwd='', db=dbfile, charset='UTF8')
    cur = con.cursor()  # 获取游标
    # cur.execute('select * from train1K')
    cur.execute('select sentences from %s ' % tableName)
    resultSet = cur.fetchall()
    result=[]
    for line in resultSet:
        result.append(line[-1])
        # print(line[-1])
    cur.close()
    con.close()
    return result

def toGoodOrBad(tableName):
    dbfile = 'datamining'
    con = mysqldb.connect(host='localhost', port=3306, user='root', passwd='', db=dbfile, charset='UTF8')
    cur = con.cursor()  # 获取游标
    # cur.execute('select * from train1K')
    cur.execute('select goodOrBad from %s ' % tableName)
    resultSet = cur.fetchall()
    result=[]
    for line in resultSet:
        result.append(line[0])
        # print(line[-1])
    cur.close()
    con.close()
    return result