#######################################

#####    Vocabulary of UI tags    #####
        
#######################################

yellow = ["yellow", "orange", "gold"]
red = ["red"]
blue = ["blue", "light blue","dark blue"]
green = ["green", "darkgreen","aquamarine"]
white = ["white"]
black = ["black", "dark"]
pink = ["pink"]
brown = ["brown"]
grey = ["grey"]
music = ["music", "music player", "musicplayer", "music_player","music-player","music players","musicplayers","music_players","apple_music", "apple music","apple-music","playlist", "music app","musicapp","music_app", "music application","music_applications", "music applications","music player app","music player ui"]
ecommerce = ["ecommerce","e commerce", "e-commerce", "commerce", " e_commerce", "e-shop","online_store","ikea", "eshop", "online_shop","adidas", "nike","footwear", "clothing", "clothes","webshop"]
food = ["salad","bakery", "steak", "cake","recipe","recipes","food","foods", "dessert", "juice","drink", "food_app","food app","food application", "foodapp","food_application","restaurant"]
travel = ["travel", "hiking","travelling","national_parks","national_park","travel_agency","tourist","travel_app", "tours","tourism","roadtrip","trip","vacation","travelapp","travel app","travelapps","travel application","travel applications","travel_applications"]
finance = ["bank", "finance", "banking","finances","finance_app","finance app","finance application","finance_app","finance_application","banking app","banding_app","bank application","bank app","bank_app","financial","investing","insurance","wallet"]
game = ["game","videogame","video game","games"]
health = ["health", "healthy","fitness"]
news = ["news","newspaper","news design",  "news grid",  "news list",  "newstemplate", "news app","news feed"]
sport = ["sport","sports","gym","workout","exercising","exercise","exercises"]
social=["socialnetwork", "social network", "social networking", "blog", "messenger", "facebook", 'instagram', 'dating', 'chat',"chatting"]
weather = ["weather","weather app","weather_app","temperature"]
#lifestyle = ["fashion", "furniture", "real estate", "real_estate"]
medical = [ "medical", "healthcare", "hospital", "pharmacy", "medicine", "disease", "drug", "pill", "treatment", "x-ray", "doctor", "blood pressure"]
book = ["magazine", "magazines", "reading", "bookstore", "digitalreading", "digital reading", "digital_reading", "digital bookstroe", "digital_bookstroe","digitalbookstroe", "book","books"]
landing = ["landing page", "landing pages","landingpage","landingpages","landing_pages","landing_page"]
checkout = [ "check_out", "check out", "checkout", "payment"]
signup = ["sign up","signup","sign_up","login","log in","log_in"]
profile = ["profile"]
search = ["search", "searching"]
dashboard = ["dashboard"]
list_ = ["list"]
form = ["form"]
grid = ["grid"]
chart = ["chart"]
simple = ["simple","clean", "minimal","minimalistic"]
flat = ["flat","flat design","flat_design","flatdesign","flat ui","flat_ui_design"]
gradient = ["color gradient", "color_gradient", "gradient color","gradient"]
illustration = ["illustration"]
ui = ["ui", "user interface","user-interface", "user_interface", "user interface design", "user_interface_design", "uidesign", "ui design", "ui_design", "uiuxdesign", "uxuidesign", "uiux design", "uxui design", "uiux_design", "uxui_design", "uiux-design", "uxui-design", "uiux", "uidesigner", "ui_ux", "ui.ux", "ux.ui", "uxui", "ui-ux", "ux-ui", "app-ui", "daily ui", "dailyui", "daily_ui", "30_days_of_ui", "30 days of ui"]
mobile = ["iphone", "iphonex","iphone_x", "iphone8", "iphone7", "ios_11","iphone_app","ipad_pro", "ios11","app", "application", "android app", "app-design",  "app design","app_design", "appdesign","ios app", "ios_app", "ios_design", "android_app", "app_development","mobile_application", "application_design", "mobile app","mobileapp","mobile", "mobile design", "mobiledesign", "mobile website", "mobilewebsite","mobile_web"]
website = ["website","webdesigner", "web-design", "design_for_website","websitedesign", "web app","webpage", "website", "web", "website design","webpage design",'fullscreen']
tablet = ['tablet', 'ipad', 'ipadpro', 'tablet app', 'mobile tablet illustrations', 'tablet design', 'tablet responsive design', 'global talent platform']

def categorization():
    tags = ['tablet','mobile','website','ui','illustration','gradient','flat','simple','chart','grid','form','list_','dashboard','search','profile','signup','checkout','landing','book','medical','weather','social','sport','news','health','game','finance','travel','food','ecommerce','music','grey','brown','pink','black','white','green','blue','red','yellow']
    d = {}
    for t in tags:
        d[t] = eval(t)
    return d

def type__():
    t = {'plat':['mobile','website','tablet'],
            'color':['grey','brown','pink','black','white','green','blue','red','yellow'],
            'function':['book','medical','weather','social','sport','news','health','game','finance','travel','food','ecommerce','music'],
            'screen_fuc':['search','profile','signup','checkout','landing'],
            'screen_lay':['chart','grid','form','list_','dashboard']}
    return t