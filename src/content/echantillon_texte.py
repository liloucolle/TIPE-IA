from datasets import load_dataset
import re

# Chargement du dataset Wikitext en mode cache réutilisable.
# Cela évite de retélécharger les mêmes données si elles sont déjà présentes.
dataset = load_dataset("wikitext", "wikitext-103-raw-v1", download_mode="reuse_cache_if_exists")
dataset = dataset["train"]  # On utilise la split d'entraînement uniquement

# Exemples de sous-ensembles de texte pour des tests ou des mesures de performance
texte_10 = dataset.select(range(10))
texte_100 = dataset.select(range(100))
texte_1000 = dataset.select(range(1000))
texte_10000 = dataset.select(range(10000))

# Article d'exemple utilisé par l'extraction de mots-clés
article = [
    "Valkyria Chronicles III , commonly referred to as Valkyria Chronicles III outside Japan , is a tactical role @-@ playing video game developed by Sega and Media.Vision for the PlayStation Portable .",
    "Released in January 2011 in Japan , it is the third game in the Valkyria series .",
    "Employing the same fusion of tactical and real @-@ time gameplay as its predecessors , the story follows the ' Nameless ' , a penal military unit serving the nation of Gallia during the Second Europan War .",
    "The game 's opening theme was sung by May .",
    "After release , it received downloadable content , along with an expanded edition in November of that year .",
    "An unofficial fan translation patch began development in February 2012 , and the patch was released in January 2014 .",
    "As with previous Chronicles games , the player progresses through a series of linear missions , gradually unlocked as maps that can be freely scanned through and replayed .",
    "Outside missions , the player characters rest in a camp , where units can be customized and character growth occurs .",
    "The game 's battle system is carried over directly from Chronicles .",
    "During missions , players select each unit using a top @-@ down perspective of the battlefield map , then move the character around the battlefield in third @-@ person .",
    "Troops are divided into five classes : Scouts , Engineers , Lancers and Armored Soldier .",
    "With victory in battle , experience points are awarded to the squad ."
]

# Articles synthétiques pour le TIPE (vocabulaire limité au corpus Wikitext)
articles = [
    "Gallia military unit perform secret operation during the Second Europan War against Imperial unit Calamaty Raven . Tactical operation follows penal military unit and black operations during war .\n\nThe military unit serving Gallia perform real time tactical operations . Secret black operations during the war against Imperial unit and military operations .",
    "The game employs tactical and real time gameplay . The story follows the same gameplay as the first game in the Valkyria series .\n\nTactical role playing video game for the PlayStation Portable follows the same fusion of gameplay and story during military operations . The game employs tactical operations during war .",
    "The Nameless are a penal military unit serving the nation of Gallia during the war . The unit perform black operations and secret operations .\n\nThe penal military unit follows tactical operations during the Second Europan War . Gallia military operations against Imperial unit operations during the war .",
    "The video game developed by Sega and Media Vision for the PlayStation Portable was released in Japan in January 2011 .\n\nThe third game in the Valkyria series follows tactical role playing gameplay . The Japanese game employs real time gameplay and tactical operations during the story .",
    "Valkyria of the Battlefield follows military unit during the Second Europan War . The story runs parallel to the first game .\n\nThe military story follows Gallia operations against Imperial operations during war . Tactical gameplay and real time gameplay are employed during battlefield operations .",
    "Imperial unit and Gallia military operations are pitted against each other during war . Secret operations perform tactical black operations .\n\nThe Imperial military unit follows tactical operation during the Second Europan War . Gallia operations and black operations during secret military gameplay .",
    "The PlayStation Portable game employs tactical role playing gameplay and real time operations . The game is the third game in the series .\n\nThe tactical game developed for Portable gameplay follows military story and secret operations during war . The game employs tactical operations against Imperial unit operations .",
    "The story runs parallel to the first Valkyria game . Military unit serving Gallia perform secret black operations during the war .\n\nParallel military operations during the Second Europan War against Imperial operations . Tactical gameplay follows military unit and penal operations during war .",
    "The Japanese game released in January 2011 outside Japan follows the same tactical gameplay as its predecessors .\n\nThe game developed by Sega and Media Vision employs tactical role playing gameplay during military operations . The story follows real time gameplay during war .",
    "Unrecorded Chronicles is the third game in the Valkyria series . The game follows military gameplay and tactical operations during war .\n\nThe story follows military unit operations during the Second Europan War . Tactical gameplay and real time gameplay are employed during secret operations against Imperial unit operations ."
]
mots_representatifs = {
    1: ["military", "operation", "Gallia", "Imperial", "war"],
    2: ["gameplay", "tactical", "real", "story", "series"],
    3: ["Nameless", "penal", "serving", "Gallia", "secret"],
    4: ["Sega", "PlayStation", "Portable", "Japan", "released"],
    5: ["Battlefield", "Valkyria", "Europan", "parallel", "story"],
    6: ["Imperial", "operations", "tactical", "Gallia", "black"],
    7: ["Portable", "role", "playing", "tactical", "third"],
    8: ["parallel", "Valkyria", "military", "black", "operations"],
    9: ["Japanese", "predecessors", "gameplay", "Japan", "Sega"],
    10: ["Chronicles", "Valkyria", "third", "gameplay", "operations"]
}