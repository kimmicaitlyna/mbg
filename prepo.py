import re
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import nltk
nltk.download('punkt_tab')

positive_words = {
    "baik",
    "bagus",
    "keren",
    "sehat",
    "dukung",
    "bantu",
    "manfaat",
    "cerdas",
    "kualitas",
    "emas",
    "prioritas",
    "kuat",
    "investasi",
    "komitmen",
    "bangun",
    "wujud",
    "tumbuh",
    "tingkat",
    "maju",
    "benar",
    "layak",
    "efektif",
    "optimal",
    "sukses",
    "hebat",
    "mantap",
    "hasil",
    "unggul",
    "tepat",
    "efektif",
    "suka",
    "penting",
    "guna",
    "sesuai"
}

negative_words = {
    "racun",
    "korupsi",
    "masalah",
    "khawatir",
    "kasus",
    "korban",
    "stop",
    "proyek",
    "mubazir",
    "jelek",
    "buruk",
    "gagal",
    "bahaya",
    "otak",
    "curang",
    "sampah",
    "bodoh",
    "cacat",
    "rugi",
    "tolol"
}

neutral_words = {
    "mbg",
    "program",
    "anak",
    "makan",
    "sekolah",
    "presiden",
    "prabowo",
    "gibran",
    "indonesia",
    "dapur",
    "menu",
    "siswa",
    "bgn",
    "rakyat",
    "negara",
    "gizi"
}

def handle_negation(tokens):
    negation = {"tidak", "bukan", "belum", "jangan", "tanpa", "kurang", "tak"}
    result = []
    i = 0

    while i < len(tokens):
        if tokens[i] in negation and i + 1 < len(tokens):

            gabung = tokens[i] + "_" + tokens[i + 1]
            result.append(gabung)
            i += 2 
        else:
            result.append(tokens[i])
            i += 1
    
    return result

def lexicon_handle(tokens):
    negation = {"tidak", "bukan", "belum", "jangan", "tanpa", "kurang", "tak"}
    score = 0
    
    for word in tokens:
        if "_" in word:
            neg, kata = word.split("_", 1)
            
            if neg in negation:
                if kata in positive_words:
                    score -= 1
                elif kata in negative_words:
                    score += 1
        elif word in positive_words:
            score += 1
        elif word in negative_words:
            score -= 1
        elif word in neutral_words:
            score += 0
        else:
            score += 0
    
    return score

def clean_text(text):
    text = text.lower()  
    text = re.sub(r'&amp;', 'dan', text) 
    text = re.sub(r'badan gizi nasional', 'bgn', text) 
    text = re.sub(r'makan bergizi gratis', 'mbg', text) 
                
    text = re.sub(r'([a-zA-Z]+)2([a-zA-Z]*)', r'\1 \1\2', text) 
    text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text) 
    text = re.sub(r'https?://\S+|www\.\S+', '', text) 
    text = re.sub(r'@\w+', '', text) 
    text = re.sub(r'#\w+', '', text) 
    text = re.sub(r'[^a-zA-Z\s]', ' ', text) 
    text = re.sub(r'\b[a-zA-Z]{1}\b', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text.strip()

normalization_dict = {
    "yg": "yang",
    "ga": "tidak",
    "gak": "tidak",
    "gk": "tidak",
    "g": "tidak",
    "nggak": "tidak",
    "ngga": "tidak",
    "engga": "tidak",
    "kaga": "tidak",
    "tdk": "tidak",
    
    "aja": "saja",
    "doang": "saja",
    "kalo": "kalau",
    "klo": "kalau",
    "tp": "tapi",
    "jd": "jadi",
    "jg": "juga",
    "dr": "dari",
    "dri": "dari",
    "dgn": "dengan",
    "dg": "dengan",
    "krn": "karena",
    "karna": "karena",
    "utk": "untuk",
    "pd": "pada",
    "sbg": "sebagai",
    "dlm": "dalam",
    
    "udah": "sudah",
    "udh": "sudah",
    "sdh": "sudah",
    "dah": "sudah",
    
    "blm": "belum",
    
    "hrs": "harus",
    "bgt": "banget",
    "bnyk": "banyak",
    "trs": "terus",
    "trus": "terus",
    "bs": "bisa",
    "lg": "lagi",
    "msh": "masih",
    "drpd": "daripada",
    "pdhl": "padahal",
    
    "knp": "kenapa",
    "gmn": "bagaimana",
    
    "skrg": "sekarang",
    "tau": "tahu",
    "liat": "lihat",
    "dpt": "dapat",
    "dapet": "dapat",
    "pake": "pakai",
    
    "bikin": "buat",
    "ngasih": "beri",
    
    "kayak": "seperti",
    "gini": "begini",
    "gitu": "begitu",
    "sampe": "sampai",
    "emang": "memang",
    "bener": "benar",
    "mending": "lebih baik",
    
    "gue": "saya",
    "gw": "saya",
    "w": "saya",
    "gua": "saya",
    "lu": "kamu",
    "lo": "kamu",
    
    "org": "orang",
    "duit": "uang"
    }

stopwords_dict = [
    "saja","ya","kan","tuh","nih","sih","kok","lah","mah","nya","dong","deh"
]

important_dict = [
    "tidak", "belum", "bukan", "jangan", "tanpa", "harus", "bisa", "dapat", "ingin", "mengapa", "kenapa", "kurang", "guna"
]


#setup stemmer
factory_stem = StemmerFactory()
stemmer = factory_stem.create_stemmer()

#setup stopword
factory_stop = StopWordRemoverFactory()
stopword_list = factory_stop.get_stop_words()
stopword_list = stopword_list + stopwords_dict

for word in important_dict:
    if word in stopword_list:
        stopword_list.remove(word)

def prepro_sentimen(text): 
    #tokenisasi
    tokens = word_tokenize(text)

    #normalisasi
    tokens = [normalization_dict[word] if word in normalization_dict else word for word in tokens]
    
    #stemming
    tokens = [stemmer.stem(word) for word in tokens]

    # negasi handling
    tokens = handle_negation(tokens)
    
    #stopword removal
    tokens = [word for word in tokens if word not in stopword_list]
    
    return " ".join(tokens)
