# %%
import json
from bs4 import BeautifulSoup
from bs4.element import Tag
import text_unify
import re

# %%
def unify_text(text):
    ntext = re.sub(r"[\.,\?]+$-", "", text)
    ntext = ntext.replace(",", " ").replace(".", " ") \
        .replace(";", " ").replace("“", " ") \
        .replace(":", " ").replace("”", " ") \
        .replace('"', " ").replace("'", " ") \
        .replace("!", " ").replace("?", " ") \
        .replace("[", " ").replace("]", " ").replace(",", " ")
    
    ntext = ' '.join(ntext.split())
    ntext = ntext.lower()
    ntext = text_unify.unify(ntext)
    return ntext

# %% [markdown]
# ######
# ***
# ### Converter functions
# ***

# %% [markdown]
# #### Doccano To Html

# %%
def labels_sort_func(label):
    return label[0]

# %%
#### Doccano To Html

def doccano2html(doccano_annot_str):
    doccano_annot = json.loads(doccano_annot_str)
    text_label = ''
    text = doccano_annot['text']
    labels = doccano_annot['labels']
    labels.sort(reverse = True, key = labels_sort_func)
    
    #have been not annotated yet
    if len(labels) == 0:
        return text_label
    
    #convertation
    text_label = text
    for label_elem in labels:
        sIndex, eIndex, label = label_elem
        value = text_label[sIndex:eIndex].strip()
        if value == '':
            continue
        text_prefix = text_label[:sIndex]
        text_label = text_prefix + ' <' + label + '>' + value + '</'+ label + '> ' + text_label[eIndex:].strip()
    
    text_label = (' '.join(text_label.split()))
    text_label = text_unify.unify(text_label)
    return text_label

# %% [markdown]
# #### Html to doccano

# %%
#### Html to doccano

def html_to_doccano(html_annot):
    soup = BeautifulSoup(html_annot,  "html.parser")
    title_text = soup.text
    
    labels = []
    pre_index = 0

    for c in soup:
        #print("//" + c.string.strip())
        #there is space -> skip
        token = c.string.strip()
        if isinstance(token, str) == False or token == None or token == '':
            continue
        if (type(c) == Tag):
            start_offset = title_text.find(token, pre_index)
            if (start_offset == -1):
                print("#there's something wrong: " + c.string)
                print("#title: " + title_text)
                continue

            end_offset = start_offset + len(token)
            labels.append([start_offset, end_offset, c.name])
            pre_index = end_offset

    doccano_format = {"text": title_text, "labels":labels}
    
    return json.dumps(doccano_format, ensure_ascii=False) + '\n'

# %% [markdown]
# #### Html to Fully

# %%
#### Html to Fully

# convert html to fully
# {"content": text_string, "annotation": [{"label":[label], "points": [{"start":start_index,"end":end_index,"text":token}]}, ...]}

def html2fully(html_annot):
    soup = BeautifulSoup(html_annot,  "html.parser")
    title_text = soup.text

    annotation = []
    pre_index = 0

    for c in soup:
        #there is space -> skip
        token = c.string.strip()
        if isinstance(token, str) == False or token == None or token == '':
            continue
        if (type(c) == Tag):
            label = c.name.lower()
        else:
            label = 'None'

        tokens = token.split()
        for itr, word in enumerate(tokens):
            if label == 'None':
                BIOE = 'O'
            elif itr == 0:
                BIOE = 'B'
            elif itr == len(tokens) -1:
                BIOE = 'E'
            else:
                BIOE = 'I'
               
            start_index = title_text.find(word, pre_index)
            if (start_index == -1):
                print("#there's something wrong: " + word)
                print("#title: " + title_text)
                continue

            end_index = start_index + len(word) - 1
            annotation.append({"label":[label, BIOE], "points": [{"start":start_index,"end":end_index,"text":word}]})
            pre_index = end_index

    fully_format = {"content": title_text, "annotation":annotation}
    return json.dumps(fully_format, ensure_ascii=False) + '\n'

# %% [markdown]
# #### Doccano to Fully

# %%
#### Doccano to Fully

def doccano2fully(doccano_annot):
    html_annot = doccano2html(doccano_annot)
    html_annot = unify_text(html_annot)
    fully_annot = html2fully(html_annot)
    return fully_annot

# %%


# %%


# %% [markdown]
# ######
# ***
# ### Extracting
# *** 

# %%


# %%


# %% [markdown]
# ######
# ***
# ### Statistic
# ***

# %%


# %%


# %%


# %% [markdown]
# ######
# ***
# ### Test
# ***

# %%
text = '{"id": 2903, "text": "[31][1428]dây đồng hồ 20mm, dây thép không gỉ 3 mắt dành cho gear active, gear sport, gear s2 classic, galaxy watch 42mm", "meta": {}, "annotation_approver": null, "comments": [], "labels": [[10, 21, "Type"], [22, 26, "NumUnit"], [28, 45, "Material"], [46, 51, "NumUnit"], [51, 120, "Target"]]}'

# %%
doccano2html(text)

# %%
doccano2html(text)

# %%
unify_text(doccano2html(text))

# %%
doccano2fully(text)


