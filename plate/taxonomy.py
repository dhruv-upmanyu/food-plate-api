"""The dishes the plate pipeline recognises, with serving sizes and where their nutrition comes from.

Each dish's nutrition (per 100 g) is built by plate/build_nutrition.py:
  * `indb`: regex over Indian Nutrient Databank recipe names; the median of all matching recipes that pass the
    consistency checks is used (one recipe's error can't drive a dish's value);
  * `usda`: USDA FoodData Central ids (FNDDS / SR Legacy), used when INDB has nothing usable.

Serving sizes are household measures (1 katori ~ 150 g of dal/sabzi, 1 medium roti ~ 40 g, ...), following the
portion conventions of the ICMR-NIN Dietary Guidelines for Indians (2024) where they apply. `countable` dishes are
portioned by counting the detected pieces (rotis, idlis, eggs); the others by serving size and visible size.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Dish:
    key: str
    name: str
    group: str
    unit: str            # what one serving looks like
    unit_g: float        # grams in one serving
    countable: bool = False
    indb: str | None = None              # regex over INDB recipe names (case-insensitive)
    usda: tuple[int, ...] = field(default_factory=tuple)  # FoodData Central ids


D = Dish
DISHES: list[Dish] = [
    # ---- breads
    D("roti", "Roti / chapati", "bread", "1 medium roti", 40, True, r"^Chapati/Roti$", (2707713,)),
    D("paratha", "Plain paratha", "bread", "1 paratha", 70, True, r"^(Plain|Tandoori) parantha", (2707715,)),
    D("stuffed_paratha", "Stuffed paratha (aloo, gobi, paneer...)", "bread", "1 paratha", 110, True,
      r"^(Potato|Radish|Cauliflower|Dal|Pea|Paneer|Methi|Onion-green chilli) parantha"),
    D("naan", "Naan", "bread", "1 naan", 90, True, r"^Naan$", (2707613,)),
    D("puri", "Puri", "bread", "1 puri", 25, True, None, (2707714,)),
    D("bhatura", "Bhatura", "bread", "1 bhatura", 80, True, None, (2707714,)),
    D("thepla", "Thepla", "bread", "1 thepla", 40, True, r"thepla"),
    D("bread_toast", "Bread / toast", "bread", "1 slice", 30, True, None, (172686,)),
    D("sandwich", "Sandwich", "bread", "1 sandwich (2 slices)", 130, True, r"sandwich \(toasted\)|^(Cucumber|Cheese and tomato|Vegetarian club) sandwich"),
    D("kaathi_roll", "Kathi roll / wrap", "bread", "1 roll", 150, True, r"kaathi roll"),
    # ---- rice
    D("rice", "Plain rice", "rice", "1 katori cooked", 150, False, r"^Boiled rice", (2708411,)),
    D("pulao", "Pulao / jeera / flavoured rice", "rice", "1 katori", 150, False,
      r"^(Plain pulao|Mixed vegetable pulao|Cumin pulao|Peas pulao|Lemon rice|Mushroom pulao)"),
    D("biryani", "Biryani", "rice", "1 plate", 250, False, r"biryani", (2706538, 2708985)),
    D("fried_rice", "Fried rice", "rice", "1 plate", 200, False, r"^Chinese fried rice"),
    D("khichdi", "Khichdi", "rice", "1 katori", 200, False, r"^(Vegetable khichdi|Spinach khichri|Cracked wheat khichri|Poshtik khichdi)"),
    D("curd_rice", "Curd rice", "rice", "1 katori", 200, False, r"^Curd rice"),
    # ---- dals and legume curries
    D("dal", "Dal (yellow / tadka)", "dal", "1 katori", 150, False,
      r"^(Washed moong dal|Washed urad dal|Mixed dal|Whole moong|Whole masoor|Arhar|Split bengal gram with bottle)", (2707427,)),
    D("dal_makhani", "Dal makhani", "dal", "1 katori", 150, False, r"^Moti mahal dal"),
    D("rajma", "Rajma", "dal", "1 katori", 150, False, r"^Kidney bean curry"),
    D("chole", "Chole / chana masala", "dal", "1 katori", 150, False, r"^Chickpeas curry", (2709632,)),
    D("kadhi", "Kadhi", "dal", "1 katori", 150, False, None, (2707431,)),
    D("sambar", "Sambar", "dal", "1 katori", 150, False, r"^Sambar$"),
    D("rasam", "Rasam", "dal", "1 katori", 150, False, r"^Rasam with"),
    D("sprouts_salad", "Sprouts / chana / corn salad", "salad", "1 katori", 100, False,
      r"^(Sprouted moong salad|Sprouted moong dal chat|Mixed pulse and vegetable salad)"),
    # ---- vegetable sabzi
    D("aloo_sabzi", "Aloo sabzi", "sabzi", "1 katori", 150, False, r"^(Dry potato|Potato curry|Pea potato curry|Potato capsicum)"),
    D("gobi_sabzi", "Gobi / aloo gobi", "sabzi", "1 katori", 150, False, r"^Potato cauliflower"),
    D("bhindi", "Bhindi", "sabzi", "1 katori", 120, False, r"^(Okra/Lady's fingers fry|Stuffed okra)"),
    D("baingan", "Baingan / bharta", "sabzi", "1 katori", 150, False, r"^(Brinjal bhartha|Stuffed brinjal|Peas brinjal bhartha)"),
    D("gourd_sabzi", "Lauki / tori / tinda", "sabzi", "1 katori", 150, False, None, (2710067,)),
    D("dry_veg", "Cabbage / beans / carrot sabzi", "sabzi", "1 katori", 120, False,
      r"^(Cabbage and peas|Carrot and cabbage with coconut|Beans with coconut|Beans foogath)"),
    D("palak_saag", "Palak / saag", "sabzi", "1 katori", 150, False, r"^(Sarson ka saag|Spinach mushroom)"),
    D("mixed_veg", "Mixed veg sabzi", "sabzi", "1 katori", 150, False, r"^(Pea curry|Mushroom matar|Pea mushroom curry)", (2710067,)),
    D("soya_chunks", "Soya chunks", "sabzi", "1 katori", 150, False, r"^(Soya chunks and peas|Soya chunks korma|Soyabean curry)"),
    D("paneer_curry", "Paneer curry (shahi, kadai, matar...)", "paneer", "1 katori", 150, False,
      r"^(Paneer curry|Pea paneer curry|Shahi paneer|Paneer in butter sauce|Methi malai paneer)"),
    D("palak_paneer", "Palak paneer", "paneer", "1 katori", 150, False, r"^Spinach paneer \(Palak", (2709631,)),
    D("paneer_bhurji", "Paneer bhurji", "paneer", "1 katori", 120, False, None, (2705740,)),
    D("paneer_tikka", "Paneer pieces / tikka", "paneer", "6 pieces", 100, False, r"^Paneer shaslik", (2705740,)),
    # ---- egg and non-veg
    D("chicken_curry", "Chicken curry", "nonveg", "1 katori", 150, False, r"^(Chicken curry|Butter chicken|Chicken korma)", (2706437,)),
    D("chicken_dry", "Chicken (tandoori / tikka / grilled)", "nonveg", "4 pieces", 120, False, r"^Tandoori chicken"),
    D("egg_curry", "Egg curry", "nonveg", "1 katori (2 eggs)", 200, False, None, (2706437,)),
    D("omelette", "Omelette", "egg", "1 omelette (2 eggs)", 110, True, r"^(Plain omelette|Spanish omelette|French omelette)"),
    D("boiled_egg", "Boiled egg", "egg", "1 egg", 50, True, None, (2706867,)),
    D("egg_bhurji", "Egg bhurji", "egg", "1 serving (2 eggs)", 110, False, r"^Scrambled egg"),
    D("fish", "Fish curry / fry", "nonveg", "1 katori", 150, False, r"^(Fish curry|Bengal fish curry|Tandoori fish|Fish tikka)"),
    D("mutton", "Mutton curry", "nonveg", "1 katori", 150, False, r"^(Mutton korma|Mutton do piaza|Mutton yakhni)"),
    # ---- dairy
    D("curd", "Curd (dahi)", "dairy", "1 katori", 100, False, None, (171284,)),
    D("raita", "Raita", "dairy", "1 katori", 100, False, r"^(Cucumber raita|Tomato onion raita|Bottle gourd raita|Mint raita|Carrot raita|Spinach raita)"),
    D("milk", "Milk", "drink", "1 glass", 200, False, None, (2705385,)),
    D("buttermilk", "Buttermilk / chaas", "drink", "1 glass", 200, False, r"^Lassi \(salted\)"),
    D("lassi", "Sweet lassi", "drink", "1 glass", 250, False, r"^Sweet Lassi"),
    D("cheese", "Cheese", "dairy", "1 slice", 20, True, None, (173414,)),
    # ---- salads and fruit
    D("veg_salad", "Salad (cucumber, tomato, onion, carrot)", "salad", "1 katori", 100, False, None, (168409, 170457, 170393)),
    D("fruit_bowl", "Cut fruit / fruit bowl", "fruit", "1 bowl", 150, False, r"^Fruit salad \(Phalon"),
    D("banana", "Banana", "fruit", "1 banana", 100, True, None, (173944,)),
    D("apple", "Apple", "fruit", "1 apple", 150, True, None, (171688,)),
    D("papaya", "Papaya", "fruit", "1 bowl", 150, False, None, (169926,)),
    D("orange", "Orange / mosambi", "fruit", "1 orange", 130, True, None, (169097,)),
    D("melon", "Watermelon / muskmelon", "fruit", "1 bowl", 150, False, None, (167765, 169092)),
    D("guava", "Guava", "fruit", "1 guava", 100, True, None, (173044,)),
    D("pomegranate", "Pomegranate", "fruit", "1 bowl", 100, False, None, (169134,)),
    D("grapes", "Grapes", "fruit", "1 bowl", 100, False, None, (174683,)),
    # ---- south Indian and breakfast
    D("idli", "Idli", "breakfast", "1 idli", 40, True, r"^Idli$", (2708346,)),
    D("dosa", "Dosa (plain)", "breakfast", "1 dosa", 80, True, None, (2708347,)),
    D("masala_dosa", "Masala dosa", "breakfast", "1 dosa", 160, True, r"^Masala dosa$", (2709129,)),
    D("uttapam", "Uttapam", "breakfast", "1 uttapam", 100, True, r"^Uttapam$"),
    D("vada", "Vada", "breakfast", "1 vada", 50, True, None, (2709130,)),
    D("upma", "Upma", "breakfast", "1 katori", 150, False, r"^(Semolina upma|Vegetable upma)", (2709128,)),
    D("poha", "Poha", "breakfast", "1 katori", 150, False, r"^(Vegetable poha|Sprouted moong poha|Poha$)"),
    D("chilla", "Chilla / cheela", "breakfast", "1 chilla", 80, True, r"^(Gram flour chilla|Moong dal stuffed cheela|Poshtik chilla|Paneer stuffed cheela)"),
    D("dhokla", "Dhokla", "breakfast", "2 pieces", 60, False, r"^(Khaman|Instant dhokla|Semolina dhokla|Gram flour and semolina dhokla)"),
    D("oats", "Oats / muesli / porridge bowl", "breakfast", "1 bowl", 200, False, r"^(Oatmeal Porridge|Cornflakes with milk|Sweet wheat porridge)"),
    D("chutney", "Chutney", "condiment", "1 tablespoon", 20, False, r"^(Coconut chutney|Mint and coriander chutney|Coriander chutney|Tomato ginger chutney)"),
    D("vermicelli", "Vermicelli / seviyan", "breakfast", "1 katori", 150, False, r"^Vermicelli upma"),
    # ---- snacks and street food
    D("samosa", "Samosa", "snack", "1 samosa", 60, True, None, (2708730,)),
    D("pakora", "Pakora / bhajiya", "snack", "1 piece", 20, True, None, (2708730,)),
    D("kachori", "Kachori", "snack", "1 kachori", 60, True, None, (2708730,)),
    D("pani_puri", "Pani puri", "snack", "6 puris", 100, False, None),
    D("pav_bhaji", "Pav bhaji", "snack", "1 plate (bhaji + 2 pav)", 250, False, r"^Pav bhaji$"),
    D("chole_bhature", "Chole bhature", "snack", "1 plate", 250, False, None),
    D("namkeen", "Namkeen / bhujia / chips", "snack", "1 small bowl", 30, False, None, (167537,)),
    D("makhana", "Makhana", "snack", "1 bowl", 25, False, None, (170149,)),
    D("nuts", "Nuts (almonds, walnuts, peanuts...)", "snack", "1 handful", 30, False, None, (168599,)),
    D("roasted_chana", "Roasted chana", "snack", "1 handful", 30, False, None, (173757,)),
    D("biscuits", "Biscuits / cookies", "snack", "2 biscuits", 20, False, None, (168014,)),
    D("protein_bar", "Protein bar", "snack", "1 bar", 50, True, None, (2708126,)),
    D("momos", "Momos", "snack", "1 momo", 25, True, None, (169779,)),
    D("noodles", "Noodles / chowmein", "snack", "1 plate", 200, False, r"^Chicken chowmein"),
    D("pasta", "Pasta", "snack", "1 plate", 200, False, None),
    D("pizza", "Pizza", "snack", "1 slice", 100, True, r"^Cheese pizza", (173292,)),
    D("burger", "Burger", "snack", "1 burger", 150, True, None, (170322,)),
    D("papad", "Papad", "condiment", "1 papad", 12, True, None),
    D("pickle", "Pickle (achar)", "condiment", "1 teaspoon", 10, False, r"^(Mango pickle|Mixed vegetable pickle)"),
    # ---- sweets
    D("gulab_jamun", "Gulab jamun", "sweet", "1 piece", 40, True, None),
    D("rasgulla", "Rasgulla", "sweet", "1 piece", 40, True, r"^Rasgulla"),
    D("jalebi", "Jalebi", "sweet", "2 pieces", 50, False, None),
    D("ladoo", "Ladoo", "sweet", "1 ladoo", 35, True, r"ladoo"),
    D("barfi", "Barfi / mithai", "sweet", "1 piece", 30, True, r"burfi"),
    D("halwa", "Halwa", "sweet", "1 katori", 100, False, r"^(Semolina halwa|Carrot halwa|Moong dal halwa)"),
    D("kheer", "Kheer", "sweet", "1 katori", 150, False, r"^(Rice kheer|Vermicelli kheer|Semolina kheer|Makhana kheer)"),
    D("ice_cream", "Ice cream / kulfi", "sweet", "1 scoop", 70, False, None),
    D("cake", "Cake / pastry", "sweet", "1 slice", 70, True, None),
    D("chocolate", "Chocolate", "sweet", "1 piece", 20, False, None),
    # ---- drinks and soups
    D("chai", "Chai (milk tea)", "drink", "1 cup", 150, False, r"^Hot tea \(Garam Chai\)"),
    D("coffee", "Coffee with milk", "drink", "1 cup", 150, False, None),
    D("black_coffee", "Black coffee", "drink", "1 cup", 200, False, None),
    D("black_tea", "Black / green tea", "drink", "1 cup", 200, False, None),
    D("juice", "Juice", "drink", "1 glass", 200, False, None),
    D("smoothie", "Smoothie / milkshake", "drink", "1 glass", 250, False, r"milkshake$|^(Banana|Mango) milkshake"),
    D("protein_shake", "Protein shake", "drink", "1 shaker", 300, False, None),
    D("coconut_water", "Coconut water", "drink", "1 glass", 200, False, None),
    D("water", "Water", "drink", "1 glass", 250, False, None),
    D("soup", "Soup", "drink", "1 bowl", 200, False, r"^(Hot and sour soup|Sweet corn soup|Bottle gourd soup)"),
    D("soft_drink", "Soft drink", "drink", "1 can", 330, False, None),
]

BY_KEY = {d.key: d for d in DISHES}
KEYS = [d.key for d in DISHES]
assert len(KEYS) == len(set(KEYS)), "duplicate dish key"
