CREATE TABLE IF NOT EXISTS mainmenu (
id integer PRIMARY KEY AUTOINCREMENT,
title text NOT NULL,
url text NOT NULL
);

CREATE TABLE IF NOT EXISTS presentations (
id integer PRIMARY KEY AUTOINCREMENT,
title text NOT NULL,
creator integer NOT NULL,
picture text NOT NULL,
time integer NOT NULL,
download_path text NOT NULL,
presentation_blob BLOB
);

CREATE TABLE IF NOT EXISTS users (
id integer PRIMARY KEY AUTOINCREMENT,
name text NOT NULL,
email text NOT NULL,
psw text NOT NULL,
time integer NOT NULL
);
