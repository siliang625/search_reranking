import gc
import os
import re
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

from data_extractor import TorchDataset
import torch

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    DistilBertForMaskedLM,
)

from IPython.display import Markdown, display
import html
batch_size = 16

train_dataset = TorchDataset(
    file_name="./data/diverse.triplets.train.tsv",
    queries_path="./data/diverse.queries.all.tsv",
    passages_path="./data/diverse.passages.all.tsv",
)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

dev_dataset = TorchDataset(
    file_name="./data/diverse.triplets.dev.tsv",
    queries_path="./data/diverse.queries.all.tsv",
    passages_path="./data/diverse.passages.all.tsv",
)
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

test_dataset = TorchDataset(
    file_name="./data/diverse.triplets.test.tsv",
    queries_path="./data/diverse.queries.all.tsv",
    passages_path="./data/diverse.passages.all.tsv",
)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
tokenizer_options = {
    "return_tensors": "pt",
    "truncation": True,
    "padding": True,
    "max_length": 512,
}
maxlen = 0
dataset = test_dataset
# inputs = (
#     list(set(dataset.queries))
#     + list(set(dataset.positive_doc))
#     + list(set(dataset.negative_doc))
# )
queries = dataset.queries[:3]
documents = dataset.positive_doc[:3] + dataset.negative_doc[:3]
# for input in inputs:
#     encodings = tokenizer(
#         re.sub(r"\.\.\.\.*", " ", input),
#         return_tensors="pt",
#         truncation=True,
#         padding=True,
#         max_length=512,
#     )
#     ids, masks = encodings["input_ids"], encodings["attention_mask"]

    # if ids.shape[1] > maxlen:
    #     maxlen = ids.shape[1]
    #     print(ids.shape, tokenizer.decode(ids[0]))


# print(maxlen)
#
# queries = ('which airport is closest to london bridge?', 'which vitamin is primarily responsible for blood clotting', 'where is your perineum', 'where is whitemarsh island', 'cost of ambulance services', 'cost of a solar pool heater', 'cost for oil change mercedes', 'who is lila downs', 'who created the square deal', 'who founded goodwill industries', 'where was what about bob filmed', 'which team did jim edmonds pitch for', 'cost of a solar pool heater', 'which animals are cnidarians', 'where was david delamare born', 'where was what about bob filmed')
# pos_docs = ('London City Airport is the closest, approximately 6 miles from the city centre.', "The 'K' in vitamin K is derived from the German word koagulation.. Coagulation is the process in which blood forms clots. Vitamin K facilitates the function of several proteins, including those that are responsible for blood clot formation.It plays a vital role in cell growth and in the metabolism of bone and other tissues.itamin K and Blood Clot Formation. Blood clots through a process called the 'coagulation cascade'. It's referred to as a cascade as it involves a cascade of enzymes activating each other. Fibrin is formed at the end of the cascade.", 'That part of the floor of the PELVIS that lies between the tops of the thighs. In the male, the perineum lies between the anus and the scrotum. In the female, it includes the external genitalia. The area between the opening of the vagina and the anus in a woman, or the area between the scrotum and the anus in a man.', 'Whitemarsh Island, Georgia. Whitemarsh Island (pronounced WIT-marsh) is a census-designated place (CDP) in Chatham County, Georgia, United States. The population was 6,792 at the 2010 census. It is part of the Savannah Metropolitan Statistical Area. The communities of Whitemarsh Island are a relatively affluent suburb of Savannah.', 'What is the average cost of an ambulance ride? In Los Angeles, basic emergency ambulance transport is about $1,000 to $1,100. The cost is more like $1,200 to $1,300 for a transport that requires advanced life support. Included is the cost for paramedics and the ambulance ride itself. However, companies can charge extra for mileage, supplies and equipment.', 'If you are currently using a gas or electric pool heater, the savings from solar pool heating system will pay for itself in 2-3 years of installation. For example, the average system costs about $5,500 and it typically costs about $2,000 a year to heat a pool with gas. solar pool heater can be installed for $3,500 to $8,000. Since the sun is free, it will cost you nothing to heat your pool from April through November. Plus, there are no regular maintenance fees.', 'Change Oil and Filter for Mercedes-Benz E350 costs an average of about $137. Skip the repair shop, our certified mechanics come to you. Get a quote Â· Book an Appointment Â· Get your car fixed at your home or office.', 'Lila Downs. Ana Lila Downs SÃ¡nchez, best known as Lila Downs (born September 19, 1968 Â· ) is an Americanâ\x80\x93Mexican singer-songwriter and actress. She performs her own compositions and the works of others in multiple genres, as well as tapping into Mexican traditional and popular music.', "The Square Deal was a program started by President Theodore  Roosevelt's based on conservation of natural resources, control of  corporations, and consumer protection. Oftenâ\x80¦ referred to as the  three C's of Roosevelt's Square Deal.", 'Americaâ\x80\x99s Original Thrift. Goodwill Industries was founded in 1902 by Rev. Edgar J. Helms, a Methodist minister and early social innovator. Helms collected used household goods and clothing in wealthier areas of Boston, then trained and hired those who were poor to mend and repair the used goods.', 'What about Bob. This a howlingly funny movie filmed entirely at Smith Mountain Lake in Virginia. Smith Mountain Lake has a tourism motto: Closer than you think!!. There are some wonderful bed and breakfasts in this area of Virginia. One even in an historic downtown areas of a nearby town to Smith Mountain Lake.', 'Jim Edmonds. James Patrick Jim Edmonds (born June 27, 1970) is an American former center fielder in Major League Baseball and a current broadcaster for Fox Sports Midwest. He played for the California/Anaheim Angels, St. Louis Cardinals, San Diego Padres, Chicago Cubs, Milwaukee Brewers, and Cincinnati Reds.', 'If you are currently using a gas or electric pool heater, the savings from solar pool heating system will pay for itself in 2-3 years of installation. For example, the average system costs about $5,500 and it typically costs about $2,000 a year to heat a pool with gas. solar pool heater can be installed for $3,500 to $8,000. Since the sun is free, it will cost you nothing to heat your pool from April through November. Plus, there are no regular maintenance fees.', 'Cnidarians are a group of aquatic invertebrates that includes jellyfish, corals, sea anemones and hydras.', "David Delamare was born in Leicester, UK but has spent most of his life in Portland, Oregon where he enjoys the cloudy weather.Though he likes to travel, he has never driven a car. He sleeps late and works deep into the night.When he's not attending films, plays, or concerts he can usually be found at home or strolling in Portland's Hawthorne District.hough he likes to travel, he has never driven a car. He sleeps late and works deep into the night. When he's not attending films, plays, or concerts he can usually be found at home or strolling in Portland's Hawthorne District.", 'What about Bob. This a howlingly funny movie filmed entirely at Smith Mountain Lake in Virginia. Smith Mountain Lake has a tourism motto: Closer than you think!!. There are some wonderful bed and breakfasts in this area of Virginia. One even in an historic downtown areas of a nearby town to Smith Mountain Lake.')
# neg_docs = ('Hilton Garden Inn London Heathrow Airport added 123 new photos to the album: HGI London Heathrow Airport Exclusive Launch Event â\x80\x94 at Hilton Garden Inn London Heathrow Airport.', 'How HIV is Transmitted. HIV is spread by sexual contact with an infected person, by sharing needles and/or syringes (primarily for drug injection) with someone who is infected, or, less commonly (and now very rarely in countries where blood is screened for HIV antibodies), through transfusions of infected blood or blood clotting factors.', 'Personal injury. A common cause associated with nonischemic priapism â\x80\x94 a persistent erection caused by excessive blood flow in the penis â\x80\x94 is trauma or injury to your genitals, pelvis or perhaps the perineum, the region involving the base of the penis and the anus.', '1 Liberty Island, exclave of New York with surrounding waters in New Jersey. 2  Shooters Island Island in the middle of Kill Van Kull, part in New Jersey and part in New York.  Plum Island, Sandy Hook Bay.', "Also, Is Your Number Up A request from area fire departments, law enforcement and ambulance services - PUT YOUR PROPER ADDRESS NUMBER UP AT YOUR RESIDENCE OR BUSINESS Check it out! The Nicholas County Clerk's office is now offering a new convenience for anyone who spends a lot of time in the records room.", "Solar thermal (ST) is one of the most cost-effective renewable energy systems. Solar thermal water heating systems collect the sun's energy in the form of thermal or heat energy. The system can save a major portion of your utility bill. This solar thermal system will cost about $4,600 US (with the price decreasing all the time). 2  A solar pool heater, popular and practical, is an open loop system. 3  It's called this because water circulates back into the pool, which is (of course) an open system.", "BMW is not all top-end cars unfortunately for you amazinBimmer. BMW doesn't have competitors to the Mercedes SL, S and CL 600 models, not to mention the 65 AMG's that cost over $195K a pop. Audi is the number three brand in terms of household income behind only Porsche and Mercedes-Benz.", 'He was ridden to victory by jockey Oliver Lewis, one of thirteen African-American jockeys to compete in the race. Since then, the Kentucky Derby has been held every year at Louisvilleâ\x80\x99s Churchill Downs racetrack, making it the longest continuous held sporting event in the United States.', 'The average may be around 1000 sq.ft. but it depends on the amount of bedrooms. My listing at 4355 Nob el Drive, # 73, is way above average, especially price per sq.ft. It is 1881 sq.ft. and the price is only. $450 - 469000, an incredible deal for a beautiful townhome.BR/2BA, approx 1061 sqft units average about $403/sqft. (This one is strange because, theoretically, the price per square foot should go down as the square footage goes up.', 'A free inside look at ABM Industries salary trends. 665 salaries for 288 jobs at ABM Industries. Salaries posted anonymously by ABM Industries employees. Best Jobs in America NEW!', "The name Bob is an English baby name. In English the meaning of the name Bob is: Abbreviation of Robert. American Meaning: The name Bob is an American baby name. In American the meaning of the name Bob is: Abbreviation of Robert.German Meaning: The name Bob is a German baby name. In German the meaning of the name Bob is: Famed, bright; shining.An all-time favorite boys' name since the Middle Ages. Famous Bearers: Scottish national hero Robert the Bruce and novelist Robert Ludlum.n American the meaning of the name Bob is: Abbreviation of Robert. German Meaning: The name Bob is a German baby name. In German the meaning of the name Bob is: Famed, bright; shining. An all-time favorite boys' name since the Middle Ages.", 'Model 120, 13â\x80\x9d Pitch..............................................................................................................................Page 6. Model 120, 14â\x80\x9d Pitch..............................................................................................................................Page 7. Model 120, 15â\x80\x9d Pitch..............................................................................................................................Page 8. Model 120, 16â\x80\x9d Pitch..............................................................................................................................Page 9.', 'When people ask the question, â\x80\x9cHow much will solar panels cost,â\x80\x9d they could really be asking either, â\x80\x9cHow much does a solar panel cost,â\x80\x9d or â\x80\x9cHow much will it cost for enough solar panels to power my house?â\x80\x9d. The first question is more directly related to solar panel cost, so weâ\x80\x99ll cover that first.The answer is a little tricky because it depends on whether you are planning to buy pre-made solar panels or make them yourself. For premade solar panels, a single panel can cost about $900, or $12 per watt.A medium sized system to provide power to a small or very energy efficient house might cost $25,000 and a solar system to power a large house could cost $50,000+.he first question is more directly related to solar panel cost, so weâ\x80\x99ll cover that first. The answer is a little tricky because it depends on whether you are planning to buy pre-made solar panels or make them yourself. For premade solar panels, a single panel can cost about $900, or $12 per watt.', 'What types of animals live in a deciduous forest? Many types of animals live in a deciduous forest. Some of those animals are squirrels, deer, skunks, bears, raccoons, coyotes, and mice.', 'Childhood & Early Life. David Ortiz was born in Saint Domingo, Dominican Republic to Enrique and Angela Rosa. His father played baseball for years in Dominican pro and semipro leagues and became a source for inspiration for Ortiz.', 'Robert B. Shepard. Childhood, Family & Music: Bob Shepard was born on April 28, 1927, in Phoenix Arizona, to Chester and Dorothy Shepard. He was raised in Riverside, California, from the time of his birth to 1945. He was the eldest of four boys (Bob, Phil, Gilbert and Wayne). His father, Chester, died when Bob was 8 years old.')
#
# inputs = list(queries) + list(pos_docs) + list(neg_docs)
# empty_list = [""] * len(inputs)
#
# for i, zipped in enumerate(inputs):
#     encodings = tokenizer(
#         zipped,
#         return_tensors="pt",
#         truncation=True,
#         padding=True,
#         max_length=512,
#     )
#     print(i, encodings["input_ids"].shape, tokenizer.decode(encodings["input_ids"][0]))
model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")
model.load_state_dict(torch.load("demo_model.pt", map_location=device))
model.to(device)
model.eval()

with torch.no_grad():
    inputs = queries + documents
    # Tokenize queries and documents
    encodings = tokenizer(inputs, **tokenizer_options).to(device)
    ids, masks = encodings["input_ids"], encodings["attention_mask"]

    # Inference pass through model
    outputs = model.distilbert(ids, masks)
    outputs_hidden = outputs.last_hidden_state.mean(dim=1)  # [:, 0]
    vec_queries = outputs_hidden[:len(queries)].unsqueeze(1)
    vec_documents = outputs_hidden[len(queries):].unsqueeze(0)

    # Compute pairwise distances between queries and documents
    all_distances, all_indices = (vec_queries - vec_documents).norm(dim=-1).sort(dim=-1)

    # Print nicely the results
    for qdistances, dindices, query in zip(all_distances, all_indices, queries):
        # display(Markdown(f"### {query}"))
        print(query)
        for dist, dindex in zip(qdistances, dindices):
            #display(Markdown(f"**{dist:.2f}**: {documents[dindex]}"))
            print(dist.item(), documents[dindex])