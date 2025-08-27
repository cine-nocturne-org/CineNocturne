import mysql.connector
from mysql.connector import Error

try:
    # Connexion aux bases de données source et cible
    conn_source = mysql.connector.connect(
        host="127.0.0.1",
        user="root",
        password="Marley08092022!",
        database="platforms"
    )
    conn_target = mysql.connector.connect(
        host="127.0.0.1",
        user="root",
        password="Marley08092022!",
        database="movies"
    )

    if conn_source.is_connected() and conn_target.is_connected():
        print("Connexions réussies à 'platforms' et 'movies'")

        # Tables à transférer
        tables = ["netflix", "prime", "hulu", "hbo", "apple"]

        for table in tables:
            cursor_source = conn_source.cursor()
            cursor_target = conn_target.cursor()

            # Vérifier si la colonne "plateform" ou "platform" existe dans la table source
            cursor_source.execute(f"DESCRIBE {table}")
            columns = [column[0] for column in cursor_source.fetchall()]

            if "plateform" in columns:
                source_plateform_column = "plateform"
            elif "platform" in columns:
                source_plateform_column = "platform"
            else:
                print(f"Erreur : la colonne 'plateform' ou 'platform' est absente dans la table '{table}'")
                continue

            # Supprimer la table si elle existe déjà dans la base de données cible
            cursor_target.execute(f"DROP TABLE IF EXISTS `{table}`")
            conn_target.commit()

            # Créer la table dans la base de données cible
            create_table_query = f"""
                CREATE TABLE `{table}` (
                    platform_id VARCHAR(20) PRIMARY KEY,
                    title VARCHAR(255),
                    genres VARCHAR(255),
                    releaseYear INT,
                    Rating FLOAT,
                    Countries TEXT,
                    platform VARCHAR(255)
                );
            """
            cursor_target.execute(create_table_query)
            conn_target.commit()
            print(f"Table {table} recréée dans 'movies'.")

            # Extraction des données de la table source
            select_query = f"SELECT platform_id, title, genres, releaseYear, Rating, Countries, {source_plateform_column} FROM {table}"
            cursor_source.execute(select_query)
            data = cursor_source.fetchall()

            # Préparer la requête d'insertion dans la table cible
            insert_query = f"""
                INSERT INTO `{table}` (platform_id, title, genres, releaseYear, Rating, Countries, platform) 
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """

            # Exécuter l'insertion des données uniquement si elles existent
            if data:
                cursor_target.executemany(insert_query, data)
                conn_target.commit()
                print(f"Données transférées pour la table {table}")
            else:
                print(f"Aucune donnée trouvée pour la table {table}")

except Error as e:
    print(f"Erreur : {e}")

finally:
    # Fermeture des connexions
    if conn_source.is_connected():
        cursor_source.close()
        conn_source.close()
        print("Connexion fermée pour 'platforms'")
    if conn_target.is_connected():
        cursor_target.close()
        conn_target.close()
        print("Connexion fermée pour 'movies'")
