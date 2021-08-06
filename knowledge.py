import wikipedia
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore",category=UserWarning)
warnings.filterwarnings("ignore",category=RuntimeWarning)
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from gensim.models import LsiModel
import gensim.models as models
import gensim.corpora as corpora
import networkx as nx
from bokeh.transform import linear_cmap
from bokeh.plotting import figure, show
from bokeh.models import Range1d, MultiLine, Circle, HoverTool, WheelZoomTool, ResetTool, SaveTool, TapTool, OpenURL
from bokeh.models.graphs import from_networkx, NodesAndLinkedEdges, EdgesAndLinkedNodes
from bokeh.palettes import cividis
from concurrent.futures import ThreadPoolExecutor
from random import shuffle

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
info_nodes = {}
word_tokens1 = []
bigram = models.Phrases(word_tokens1, min_count=5, threshold=50)
bigram_mod = models.phrases.Phraser(bigram)
filler = stopwords.words("english")
filler.extend(["to", "a", ".the", "it", "on", "===", "==", "``", "\\'\\'", "'s", "''", "...", "====", "-the", "the", "in"])
filler = set(filler)
punc = set((string.punctuation))
queryPage = None
i = 1

def collectPage(n):
    global queryPage
    try:
        queryPage = wikipedia.page(n.lower())
    except wikipedia.exceptions.DisambiguationError as e:
        s = e.options
        for a in s:
            try:
                queryPage = wikipedia.page(a.lower())
                break
            except:
                continue
    except Exception as e:
        print(type(e))
        return None
    queryContent = queryPage.content
    word_tokens2 = word_tokenize(queryContent)
    word_tokens1.extend(list(word_tokens2))
    filtered_content = [word for word in word_tokens2 if word.lower() not in filler]
    filtered_content = [word for word in filtered_content if word not in punc]
    filtered_content = [word for word in filtered_content if word.isnumeric() == False]
    initial_doc = [stemmer.stem(lemmatizer.lemmatize(w)) for w in filtered_content]
    bigram_mod[initial_doc]
    return queryPage.title, initial_doc

query = input("What Knowledge would you like to know: ")
connection_percentage = int(input("What should the minimum correlation be to original topic (Below 100): "))
query, initial_doc = collectPage(query)

try:
    print(len(queryPage.links))
    info_nodes.update({query : queryPage.links})
    max_docs = int(input("What are the maximum amount of docs you would like to have: "))
except:
    print("Your code has failed (Probably because there's a problem with internet).")

def correlation_loop(a):
    global i
    if i >= max_docs:
        return
    try:
        queryPage = wikipedia.page(a.lower())
    except wikipedia.exceptions.DisambiguationError as e:
        s = e.options
        for a in s:
            try:
                queryPage = wikipedia.page(a)
                break
            except:
                continue
    except Exception as e:
        return
    links = queryPage.links
    percentage = int(len(set(info_node).intersection(links))/len(info_node) * 100)
    if percentage >= connection_percentage and percentage < 100:
        info_nodes.update({a : links})
        i += 1
        return
    else:
        return

info_node = info_nodes[query]
shuffle(info_node)
batchsize = 20

for a in range(0, len(info_node), batchsize):
    batch = info_node[a:a+batchsize]
    with ThreadPoolExecutor(max_workers=8) as executor:
        nodes = executor.map(correlation_loop, batch)
    if i >= max_docs:
        break
    print(str((i/max_docs) * 100) + "% Completed")

print(info_nodes.keys())
g = nx.from_dict_of_lists(info_nodes)
pos = nx.spring_layout(g)
degree_list = nx.degree_centrality(g)
degree_list2 = sorted(degree_list.items(), key=lambda x: x[1])
important_docs = [degree[0] for degree in degree_list2 if degree[1] > (sum(degree_list.values())/len(degree_list))*10]
print("We're not done yet, so just be patient")
def New_graph(docs1):
    new_nodes = {a : list(set(docs1).intersection(list(g.neighbors(a)))) for a in docs1}
    g2 = nx.from_dict_of_lists(new_nodes)
    pos2 = nx.spring_layout(g2)
    return g2, pos2
def determine_score(tuple):
    query, doc = collectPage(tuple[0])
    ldamodel = tuple[1]
    dictionary = tuple[2]
    try:
        bow_vector = dictionary.doc2bow(doc)
    except Exception as e:
        print(type(e))
        return None
    score_list = [score for index, score in list(ldamodel[bow_vector])]
    score = sum(score_list)/len(score_list)
    print(score)
    return score
def Modelling(initial_doc):
    dataset = [d.split() for d in initial_doc]
    dictionary = corpora.Dictionary(dataset)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in dataset]
    ldamodel = LsiModel(corpus=doc_term_matrix, id2word=dictionary)
    return ldamodel, dictionary

w = query
y = important_docs
z = initial_doc
def html_network():
    g2, pos2 = New_graph(y)
    ldamodel, dictionary = Modelling(z)
    model = [ldamodel] * len(y)
    dictionary = [dictionary] * len(y)
    zipped_results = list(zip(y, model, dictionary))

    with ThreadPoolExecutor(max_workers=8) as executor:
        results2 = executor.map(determine_score, zipped_results)
    results2 = list(results2)

    plot = figure(sizing_mode="stretch_both", x_range=Range1d(-1.1, 1.1), y_range=Range1d(-1.1, 1.1), tools="tap")
    plot.axis.visible = False
    plot.xgrid.visible = False
    plot.ygrid.visible = False
    plot.title.text =  w + " Knowledge Graph"

    node_hover_tool = HoverTool(tooltips=[("Wikipedia page", "@index")])
    plot.add_tools(node_hover_tool, WheelZoomTool(), ResetTool(), SaveTool())
    graph_renderer = from_networkx(g2, pos2, scale=1, center=(0, 0))
    if 0 < len(results2) <= 256:
        palette = cividis(len(results2))
    elif len(results2) > 256:
        palette = cividis(256)

    graph_renderer.node_renderer.data_source.data["color"] = results2
    mapper = linear_cmap(field_name="color", palette=palette, low=min(results2), high=max(results2), nan_color="white")
    graph_renderer.node_renderer.glyph = Circle(size=15, fill_color=mapper)
    graph_renderer.edge_renderer.glyph = MultiLine(line_color="#CCCCCC", line_alpha=0.8, line_width=5)
    graph_renderer.edge_renderer.hover_glyph = MultiLine(line_color="black", line_width=5)

    graph_renderer.inspection_policy = NodesAndLinkedEdges()

    plot.renderers.append(graph_renderer)

    url = "https://en.wikipedia.org/wiki/@index"
    taptool = plot.select(type=TapTool)
    taptool.callback = OpenURL(url=url)

    show(plot)

html_network()
