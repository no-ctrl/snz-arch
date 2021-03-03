import re


def get_words(doc):
    """Поделба на документот на зборови. Стрингот се дели на зборови според
    празните места и интерпукциските знаци

    :param doc: документ
    :type doc: str
    :return: множество со зборовите кои се појавуваат во дадениот документ
    :rtype: set(str)
    """
    # подели го документот на зборови и конвертирај ги во мали букви
    # па потоа стави ги во резултатот ако нивната должина е >2 и <20
    words = set()
    for word in re.split('\\W+', doc):
        if 2 < len(word) < 20:
            words.add(word.lower())
    return words
def get_words_with_ignore(doc):
    """Поделба на документот на зборови. Стрингот се дели на зборови според
    празните места и интерпукциските знаци

    :param doc: документ
    :type doc: str
    :return: множество со зборовите кои се појавуваат во дадениот документ
    :rtype: set(str)
    """
    # подели го документот на зборови и конвертирај ги во мали букви
    # па потоа стави ги во резултатот ако нивната должина е >2 и <20
    words = get_words()
    for word in re.split('\\W+', doc):
        if 2 < len(word) < 20 :
            if word in words_to_ignore:
                words.add(word.lower())
    return words


def get_words_with_ignore(doc):
    """Поделба на документот на зборови. Стрингот се дели на зборови според
    празните места и интерпукциските знаци

    :param doc: документ
    :type doc: str
    :return: множество со зборовите кои се појавуваат во дадениот документ
    :rtype: set(str)
    """
    # подели го документот на зборови и конвертирај ги во мали букви
    # па потоа стави ги во резултатот ако нивната должина е >2 и <20
    words = set()
    for word in re.split('\\W+', doc):
        if 2 < len(word) < 20:
            if word not in words_to_ignore:
                words.add(word.lower())
    return words


class DocumentClassifier:
    def __init__(self, get_features):
        # број на парови атрибут/категорија (feature/category)
        self.feature_counts_per_category = {}
        # број на документи во секоја категорија
        self.category_counts = {}
        # функција за добивање на атрибутите (зборовите) во документот
        self.get_features = get_features

    def increment_feature_counts_per_category(self, current_feature, current_category):
        """Зголемување на бројот на парови атрибут/категорија

        :param current_feature: даден атрибут
        :param current_category: дадена категорија
        :return: None
        """
        self.feature_counts_per_category.setdefault(current_feature, {})
        self.feature_counts_per_category[current_feature].setdefault(current_category, 0)
        self.feature_counts_per_category[current_feature][current_category] += 1

    def increment_category_counts(self, cat):
        """Зголемување на бројот на предмети (документи) во категорија

        :param cat: категорија
        :return: None
        """
        self.category_counts.setdefault(cat, 0)
        self.category_counts[cat] += 1

    def get_feature_counts_per_category(self, current_feature, current_category):
        """Добивање на бројот колку пати одреден атрибут се има појавено во
        одредена категорија

        :param current_feature: атрибут
        :param current_category: категорија
        :return: None
        """
        if current_feature in self.feature_counts_per_category \
                and current_category in self.feature_counts_per_category[current_feature]:
            return float(self.feature_counts_per_category[current_feature][current_category])
        return 0.0

    def get_category_count(self, current_category):
        """Добивање на бројот на предмети (документи) во категорија

        :param current_category: категорија
        :return: број на предмети (документи)
        """
        if current_category in self.category_counts:
            return float(self.category_counts[current_category])
        return 0

    def get_total_count(self):
        """Добивање на вкупниот број на предмети"""
        return sum(self.category_counts.values())

    def categories(self):
        """Добивање на листа на сите категории"""
        return self.category_counts.keys()

    def train(self, item, current_category):
        """Тренирање на класификаторот. Новиот предмет (документ)

        :param item: нов предмет (документ)
        :param current_category: категорија
        :return: None
        """
        # Се земаат атрибутите (зборовите) во предметот (документот)
        features = self.get_features(item)
        # Се зголемува бројот на секој атрибут во оваа категорија
        for current_feature in features:
            self.increment_feature_counts_per_category(current_feature, current_category)

        # Се зголемува бројот на предмети (документи) во оваа категорија
        self.increment_category_counts(current_category)

    def get_feature_per_category_probability(self, current_feature, current_category):
        """Веројатноста е вкупниот број на пати кога даден атрибут f (збор) се појавил во
        дадена категорија поделено со вкупниот број на предмети (документи) во категоријата

        :param current_feature: атрибут
        :param current_category: карактеристика
        :return: веројатност на појавување
        """
        if self.get_category_count(current_category) == 0:
            return 0
        return self.get_feature_counts_per_category(current_feature, current_category) \
               / self.get_category_count(current_category)

    def weighted_probability(self, current_feature, current_category, prf, weight=1.0, ap=0.5):
        """Пресметка на тежински усогласената веројатност

        :param current_feature: атрибут
        :param current_category: категорија
        :param prf: функција за пресметување на основната веројатност
        :param weight: тежина
        :param ap: претпоставена веројатност
        :return: тежински усогласена веројатност
        """
        # Пресметај ја основната веројатност
        basic_prob = prf(current_feature, current_category)
        # Изброј колку пати се има појавено овој атрибут (збор) во сите категории
        totals = sum([self.get_feature_counts_per_category(current_feature, currentCategory) for currentCategory in
                      self.categories()])
        # Пресметај ја тежински усредената веројатност
        bp = ((weight * ap) + (totals * basic_prob)) / (weight + totals)
        return bp


class NaiveBayes(DocumentClassifier):
    def __init__(self, get_features):
        super().__init__(get_features)
        self.thresholds = {}

    def set_threshold(self, current_category, threshold):
        """Поставување на праг на одлучување за категорија

        :param current_category: категорија
        :param threshold: праг на одлучување
        :return: None
        """
        self.thresholds[current_category] = threshold

    def get_threshold(self, current_category):
        """Добивање на прагот на одлучување за дадена класа

        :param current_category: категорија
        :return: праг на одлучување за дадената категорија
        """
        if current_category not in self.thresholds:
            return 1.0
        return self.thresholds[current_category]

    def calculate_document_probability_in_class(self, item, current_category):
        """Ја враќа веројатноста на документот да е од класата current_category
        (current_category е однапред позната)

        :param item: документ
        :param current_category: категорија
        :return:
        """
        # земи ги зборовите од документот item
        features = self.get_features(item)
        # помножи ги веројатностите на сите зборови
        p = 1
        for current_feature in features:
            p *= self.weighted_probability(current_feature, current_category,
                                           self.get_feature_per_category_probability)

        return p

    def get_category_probability_for_document(self, item, current_category):
        """Ја враќа веројатноста на класата ако е познат документот

        :param item: документ
        :param current_category: категорија
        :return: веројатност за документот во категорија
        """
        cat_prob = self.get_category_count(current_category) / self.get_total_count()
        calculate_document_probability_in_class = self.calculate_document_probability_in_class(item, current_category)
        # Bayes Theorem
        return calculate_document_probability_in_class * cat_prob / (1.0 / self.get_total_count())

    def classify_document(self, item, default=None):
        """Класифицирање на документ

        :param item: документ
        :param default: подразбирана (default) класа
        :return:
        """
        probs = {}
        # најди ја категоријата (класата) со најголема веројатност
        max = 0.0
        for cat in self.categories():
            probs[cat] = self.get_category_probability_for_document(item, cat)
            if probs[cat] > max:
                max = probs[cat]
                best = cat

        # провери дали веројатноста е поголема од threshold*next best (следна најдобра)
        for cat in probs:
            if cat == best:
                continue
            if probs[cat] * self.get_threshold(best) > probs[best]: return default

        return best


train_data = [
    ("""What Are We Searching for on Mars?
Martians terrified me growing up. I remember watching the 1996 movie Mars Attacks! and fearing that the Red Planet harbored hostile alien neighbors. Though I was only 6 at the time, I was convinced life on Mars meant little green men wielding vaporizer guns. There was a time, not so long ago, when such an assumption about Mars wouldn’t have seemed so far-fetched.
Like a child watching a scary movie, people freaked out after listening to “The War of the Worlds,” the now-infamous 1938 radio drama that many listeners believed was a real report about an invading Martian army. Before humans left Earth, humanity’s sense of what—or who—might be in our galactic neighborhood was, by today’s standards, remarkably optimistic.
""",
     "science"),
    ("""Mountains of Ice are Melting, But Don't Panic (Op-Ed)
If the planet lost the entire West Antarctic ice sheet, global sea level would rise 11 feet, threatening nearly 13 million people worldwide and affecting more than $2 trillion worth of property. 
Ice loss from West Antarctica has been increasing nearly three times faster in the past decade than during the previous one — and much more quickly than scientists predicted.
This unprecedented ice loss is occurring because warm ocean water is rising from below and melting the base of the glaciers, dumping huge volumes of additional water — the equivalent of a Mt. Everest every two years — into the ocean.
""",
     "science"),
    ("""Some scientists think we'll find signs of aliens within our lifetimes. Here's how.
Finding extraterrestrial life is the essence of science fiction. But it's not so far-fetched to predict that we might find evidence of life on a distant planet within a generation.
"With new telescopes coming online within the next five or ten years, we'll really have a chance to figure out whether we're alone in the universe," says Lisa Kaltenegger, an astronomer and director of Cornell's new Institute for Pale Blue Dots, which will search for habitable planets. "For the first time in human history, we might have the capability to do this."
""",
     "science"),
    ("""'Magic' Mushrooms in Royal Garden: What Is Fly Agaric?
Hallucinogenic mushrooms are perhaps the last thing you'd expect to find growing in the Queen of England's garden.
Yet a type of mushroom called Amanita muscaria — commonly known as fly agaric, or fly amanita — was found growing in the gardens of Buckingham Palace by the producers of a television show, the Associated Press reported on Friday (Dec. 12).
A. muscaria is a bright red-and-white mushroom, and the fungus is psychoactive when consumed.
""",
     "science"),
    ("""Upcoming Parks : 'Lost Corner' Finds New Life in Sandy Springs
At the corner of Brandon Mill Road, where Johnson Ferry Road turns into Dalrymple Road, tucked among 24 forested acres, sits an early 20th Century farmhouse. A vestige of Sandy Springs' past, the old home has found new life as the centerpiece of Lost Forest Preserve. While the preserve isn't slated to officially debut until some time next year, the city has opened the hiking trails to the public until construction begins on the permanent parking lot (at the moment the parking lot is a mulched area). The new park space includes community garden plots, a 4,000-foot-long hiking trail and an ADA-accessible trail through the densely wooded site. For Atlantans seeking an alternate escape to serenity (or those who dig local history), it's certainly worth a visit.
""",
     "science"),
    ("""Stargazers across the world got a treat this weekend when the Geminids meteor shower gave the best holiday displays a run for their money.
The meteor shower is called the "Geminids" because they appear as though they are shooting out of the constellation of Gemini. The meteors are thought to be small pieces of an extinct comment called 3200 Phaeton, a dust cloud revolving around the sun. Phaeton is thought to have lost all of its gas and to be slowly breaking apart into small particles.
Earth runs into a stream of debris from 3200 Phaethon every year in mid-December, causing a shower of meteors, which hit its peak over the weekend.
""",
     "science"),
    ("""Envisioning a River of Air
By the classification rules of the world of physics, we all know that the Earth's atmosphere is made of gas (rather than liquid, solid, or plasma). But in the world of flying it's often useful to think
""",
     "science"),
    ("""Following Sunday's 17-7 loss to the Seattle Seahawks, the San Francisco 49ers were officially eliminated from playoff contention, and they have referee Ed Hochuli to blame. OK, so they have a lot of folks to point the finger at for their 7-7 record, but Hochuli's incorrect call is the latest and easiest scapegoat.
"""
     , "sport"),
    ("""Kobe Bryant and his teammates have an odd relationship. That makes sense: Kobe Bryant is an odd guy, and the Los Angeles Lakers are an odd team.
They’re also, for the first time this season, the proud owners of a three-game winning streak. On top of that, you may have heard, Kobe Bryant passed Michael Jordan on Sunday evening to move into third place on the NBA’s all-time scoring list. 
"""
     , "sport"),
    ("""The Patriots continued their divisional dominance and are close to clinching home-field advantage throughout the AFC playoffs. Meanwhile, both the Colts and Broncos again won their division titles with head-to-head wins.The Bills' upset of the Packers delivered a big blow to Green Bay's shot at clinching home-field advantage throughout the NFC playoffs. Detroit seized on the opportunity and now leads the NFC North.
"""
     , "sport"),
    ("""If you thought the Washington Redskins secondary was humbled by another scintillating performance from New Yorks Giants rookie wide receiver sensation Odell Beckham Jr., think again.In what is becoming a weekly occurrence, Beckham led NFL highlight reels on Sunday, collecting 12 catches for 143 yards and three touchdowns in Sunday's 24-13 victory against an NFC East rival. 
"""
     , "sport")
    , ("""That was two touchdowns and 110 total yards for the three running backs. We break down the fantasy implications.The New England Patriots' rushing game has always been tough to handicap. Sunday, all three of the team's primary running backs put up numbers, and all in different ways, but it worked for the team, as the Patriots beat the Miami Dolphins, 41-13.
"""
       , "sport"),
    ("""General Santos (Philippines) (AFP) - Philippine boxing legend Manny Pacquiao vowed to chase Floyd Mayweather into ring submission after his US rival offered to fight him next year in a blockbuster world title face-off. "He (Mayweather) has reached a dead end. He has nowhere to run but to fight me," Pacquiao told AFP late Saturday, hours after the undefeated Mayweather issued the May 2 challenge on US television. The two were long-time rivals as the "best pound-for-pound" boxers of their generation, but the dream fight has never materialised to the disappointment of the boxing world.
"""
     , "sport"),
    ("""When St. John's landed Rysheed Jordan, the consensus was that he would be an excellent starter.
So far, that's half true.
Jordan came off the bench Sunday and tied a career high by scoring 24 points to lead No. 24 St. John's to a 74-53 rout of Fordham in the ECAC Holiday Festival.
''I thought Rysheed played with poise,'' Red Storm coach Steve Lavin said. ''Played with the right pace. Near perfect game.''
"""
     , "sport"),
    ("""Five-time world player of the year Marta scored three goals to lead Brazil to a 3-2 come-from-behind win over the U.S. women's soccer team in the International Tournament of Brasilia on Sunday. Carli Lloyd and Megan Rapinoe scored a goal each in the first 10 minutes to give the U.S. an early lead, but Marta netted in the 19th, 55th and 66th minutes to guarantee the hosts a spot in the final of the four-team competition.
"""
     , "sport"),
]

words_to_ignore = ['and', 'are', 'for', 'was', 'what', 'when', 'who', 'but', 'from', 'after', 'out', 'our', 'my', 'the',
                   'with', 'some', 'not', 'this', 'that']

if __name__ == '__main__':
    recenica = input()
    klasifikator=NaiveBayes(get_words)
    klasifikator2=NaiveBayes(get_words_with_ignore)
    for red in train_data:
        klasifikator.train(red[0],red[1])
        klasifikator2.train(red[0],red[1])
    cl1=klasifikator.classify_document(recenica)
    cl2=klasifikator2.classify_document(recenica)
    print(cl1)
    print(cl2)
    if cl1  != cl2:
        print("kontradikcija")