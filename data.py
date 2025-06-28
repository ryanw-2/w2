import mysql.connector

db = mysql.connector.connect(
    host = "localhost",
    user = "root",
    passwd = "abcd",
    database = "w2database"
    )

mycursor = db.cursor()
# mycursor.execute("CREATE DATABASE w2database")
# mycursor.execute("CREATE TABLE Points (pointID int PRIMARY KEY NOT NULL AUTO_INCREMENT, class ENUM('DOOR', 'WIN', 'OUTERWALL') NOT NULL, x1 smallint NOT NULL, y1 smallint NOT NULL, x2 smallint NOT NULL, y2 smallint NOT NULL)")
Q1 = "CREATE TABLE WallTypes (class ENUM('DOOR', 'WIN', 'OUTERWALL') PRIMARY KEY NOT NULL, " \
"                             wall_width smallint DEFAULT 4)"

Q2 = "CREATE TABLE Points (id int PRIMARY KEY NOT NULL AUTO_INCREMENT, " \
"                          class ENUM('DOOR', 'WIN', 'OUTERWALL'), " \
"                          FOREIGN KEY(class) REFERENCES WallTypes(class), " \
"                          x1 smallint NOT NULL, " \
"                          y1 smallint NOT NULL, " \
"                          x2 smallint NOT NULL, " \
"                          y2 smallint NOT NULL)"

# mycursor.execute(Q1)
# mycursor.execute(Q2)
'''
FIRST INSERT
'''

'''
sample data
'''
walltypes_data = [('DOOR', 4), 
                  ('WIN', 3), 
                  ('OUTERWALL', 6)]

points_data = [('DOOR', 1, 2, 3, 4),
               ('WIN', 1, 4, 5, 10),
               ('OUTERWALL', 1, 2, 3, 4)]

# mycursor.execute("INSERT INTO Points (class, x1, y1, x2, y2) VALUES (%s, %s, %s, %s, %s)", ("OUTERWALL", 10, 9, 5, 5))
# db.commit()

'''
THEN SELECT TO MYCURSOR
'''
# mycursor.execute("DROP TABLE Points")
# mycursor.execute("DROP TABLE WallTypes")
# mycursor.execute("SELECT ___col___ FROM ___table___ WHERE ___col condition___ ORDER BY ___col___ ___ASC or DESC___")
# mycursor.execute("ALTER TABLE ___table_name___ ADD COLUMN ___colname___ ___type___")
# mycursor.execute("ALTER TABLE ___table_name___ CHANGE ___old_colname___ ___new_colname___ ___new_type___ ")
# mycursor.execute("SELECT class, x2 FROM Points WHERE class = 'OUTERWALL' ORDER BY pointID DESC")
QSELECT1 = "SELECT * FROM Points"
QSELECT2 = "SELECT * FROM WallTypes"

Q3 = "INSERT INTO WallTypes (class, wall_width) VALUES (%s, %s)"
# for wall_type in walltypes_data:
#     mycursor.execute(Q3, wall_type)
#     db.commit()

Q4 = "INSERT INTO Points (class, x1, y1, x2, y2) VALUES (%s, %s, %s, %s, %s)"
# for x in points_data:
#     mycursor.execute(Q4, x)
#     db.commit()

for x, wall_type in enumerate(walltypes_data):
    mycursor.execute(Q3, wall_type)
    last_id = mycursor.lastrowid
    mycursor.execute(Q4, (last_id,) + points_data[x])

db.commit()

'''
THEN LOOP AND DISPLAY
'''
# mycursor.execute("DROP TABLE Points")
# mycursor.execute("DROP TABLE WallTypes")

# mycursor.execute("SELECT * FROM Points")
# mycursor.execute("SHOW TABLES")
# # type error ignore 
# for x in mycursor:
#     print(x)
