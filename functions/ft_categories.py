

# ft_cats=[
# 'sum_extension.supervised_stance_single_ft',
# 'youtube_1hot_fts',
# 'sum_user.friends_count',
# 'twitter_1hot_fts',
# 'sum_extension.botometer.scores',
# 'sum_extension.botometer.display_scores',
# 'sum_user.favourites_count',
# 'reddit_fts',
# 'sum_extension.sentiment_scores',
# 'aux_infoID_fts',
# 'sum_user.followers_count',
# 'sum_user.listed_count',
# 'aux_twitter_user_age_fts',
# 'target_action_value',
# 'aux_twitter_action_fts',
# 'gdelt_num_mentions',
# 'sum_extension.botometer.cap',
# 'sum_extension.supervised_stance_3_fts',
# 'sum_favorite_count',
# 'infoID_1hot_fts',
# 'aux_youtube_user_age_fts',
# 'sum_extension.botometer.categories',
# 'aux_youtube_action_fts',
# 'gdelt_sentiment_fts',
# ]

extra_twitter_ft_category_order_list=[
'sum_extension.supervised_stance_single_ft',
'sum_user.friends_count',
'sum_extension.botometer.scores',
'sum_extension.botometer.display_scores',
'sum_user.favourites_count',
'sum_extension.sentiment_scores',
'sum_user.followers_count',
'sum_user.listed_count',
'sum_extension.botometer.cap',
'sum_extension.supervised_stance_3_fts',
'sum_favorite_count',
'sum_extension.botometer.categories',
]



ft_category_bool_int_dict={
'target_action_value':1,

'youtube_1hot_fts':1,
'twitter_1hot_fts':1,
'infoID_1hot_fts':1,

'reddit_fts':1,
'gdelt_num_mentions':1,
'gdelt_sentiment_fts':1,

'aux_twitter_user_age_fts':1,
'aux_youtube_user_age_fts':1,

'aux_youtube_action_fts':1,
'aux_twitter_action_fts':1,

'aux_infoID_fts':1,

'sum_extension.supervised_stance_single_ft':1,
'sum_user.friends_count':1,
'sum_extension.botometer.scores':1,
'sum_extension.botometer.display_scores':1,
'sum_user.favourites_count':1,
'sum_extension.sentiment_scores':1,
'sum_user.followers_count':1,
'sum_user.listed_count':1,
'sum_extension.botometer.cap':1,
'sum_extension.supervised_stance_3_fts':1,
'sum_favorite_count':1,
'sum_extension.botometer.categories':1



}

ft_category_order_list=[
'target_action_value',

'youtube_1hot_fts',
'twitter_1hot_fts',
'infoID_1hot_fts',

'reddit_fts',
'gdelt_num_mentions',
'gdelt_sentiment_fts',

'aux_twitter_user_age_fts',
'aux_youtube_user_age_fts',

'aux_youtube_action_fts',
'aux_twitter_action_fts',

'aux_infoID_fts',

'sum_extension.supervised_stance_single_ft',
'sum_user.friends_count',
'sum_extension.botometer.scores',
'sum_extension.botometer.display_scores',
'sum_user.favourites_count',
'sum_extension.sentiment_scores',
'sum_user.followers_count',
'sum_user.listed_count',
'sum_extension.botometer.cap',
'sum_extension.supervised_stance_3_fts',
'sum_favorite_count',
'sum_extension.botometer.categories',
]


ft_to_ft_cat_dict = {
"NumMentions" : "gdelt_num_mentions",
"AvgTone" : "gdelt_sentiment_fts",
"GoldsteinScale": "gdelt_sentiment_fts",
"reddit_post":"reddit_fts",
"reddit_comment":"reddit_fts",
"target_action_value":"target_action_value",
"arrests":"infoID_1hot_fts",
"arrests/opposition":"infoID_1hot_fts",
"guaido/legitimate":"infoID_1hot_fts",
"international/aid":"infoID_1hot_fts",
"international/aid_rejected":"infoID_1hot_fts",
"international/respect_sovereignty":"infoID_1hot_fts",
"maduro/cuba_support":"infoID_1hot_fts",
"maduro/dictator":"infoID_1hot_fts",
"maduro/legitimate":"infoID_1hot_fts",
"maduro/narco":"infoID_1hot_fts",
"military":"infoID_1hot_fts",
"military/desertions":"infoID_1hot_fts",
"other/anti_socialism":"infoID_1hot_fts",
"other/censorship_outage":"infoID_1hot_fts",
"other/chavez":"infoID_1hot_fts",
"other/chavez/anti":"infoID_1hot_fts",
"protests":"infoID_1hot_fts",
"violence":"infoID_1hot_fts",	
"reddit_post":"reddit_fts",
"reddit_comment":"reddit_fts",
"twitter_retweet":"twitter_1hot_fts",
"twitter_tweet":"twitter_1hot_fts",
"youtube_comment":"youtube_1hot_fts",
"youtube_video":"youtube_1hot_fts",
"AvgTone":"gdelt_sentiment_fts",
"GoldsteinScale":"gdelt_sentiment_fts",
"aux_twitter_retweet":"aux_twitter_action_fts",
"aux_twitter_tweet":"aux_twitter_action_fts",
"aux_youtube_video":"aux_youtube_action_fts",
"aux_youtube_comment":"aux_youtube_action_fts",
"aux_arrests":"aux_infoID_fts",
"aux_arrests/opposition":"aux_infoID_fts",
"aux_guaido/legitimate":"aux_infoID_fts",
"aux_international/aid":"aux_infoID_fts",
"aux_international/aid_rejected":"aux_infoID_fts",
"aux_international/respect_sovereignty":"aux_infoID_fts",
"aux_maduro/cuba_support":"aux_infoID_fts",
"aux_maduro/dictator":"aux_infoID_fts",
"aux_maduro/legitimate":"aux_infoID_fts",
"aux_maduro/narco":"aux_infoID_fts",
"aux_military":"aux_infoID_fts",
"aux_military/desertions":"aux_infoID_fts",
"aux_other/anti_socialism":"aux_infoID_fts",
"aux_other/censorship_outage":"aux_infoID_fts",
"aux_other/chavez":"aux_infoID_fts",
"aux_other/chavez/anti":"aux_infoID_fts",
"aux_protests":"aux_infoID_fts",
"aux_violence":"aux_infoID_fts",
"aux_num_youtube_new_users":"aux_youtube_user_age_fts",
"aux_num_youtube_old_users":"aux_youtube_user_age_fts",
"aux_num_twitter_new_users":"aux_twitter_user_age_fts",
"aux_num_twitter_old_users":"aux_twitter_user_age_fts",
'sum_extension.botometer.cap.english':"sum_extension.botometer.cap",
'sum_extension.botometer.cap.universal':"sum_extension.botometer.cap",
'sum_extension.botometer.categories.content':"sum_extension.botometer.categories",
'sum_extension.botometer.categories.friend':"sum_extension.botometer.categories",
'sum_extension.botometer.categories.network':"sum_extension.botometer.categories",
'sum_extension.botometer.categories.sentiment':"sum_extension.botometer.categories",
'sum_extension.botometer.categories.temporal':"sum_extension.botometer.categories",
'sum_extension.botometer.display_scores.english':"sum_extension.botometer.display_scores",
'sum_extension.botometer.display_scores.friend':"sum_extension.botometer.display_scores",
'sum_extension.botometer.display_scores.network':"sum_extension.botometer.display_scores",
'sum_extension.botometer.display_scores.sentiment':"sum_extension.botometer.display_scores",
'sum_extension.botometer.display_scores.temporal':"sum_extension.botometer.display_scores",
'sum_extension.botometer.display_scores.universal':"sum_extension.botometer.display_scores",
'sum_extension.botometer.display_scores.user':"sum_extension.botometer.display_scores",
'sum_extension.botometer.scores.english':"sum_extension.botometer.scores",
'sum_extension.botometer.scores.universal':"sum_extension.botometer.scores",
'sum_extension.sentiment_scores.negative':"sum_extension.sentiment_scores",
'sum_extension.sentiment_scores.neutral':"sum_extension.sentiment_scores",
'sum_extension.sentiment_scores.positive':"sum_extension.sentiment_scores",
'sum_extension.supervised_stance':'sum_extension.supervised_stance_single_ft',
'sum_extension.supervised_stance.?':"sum_extension.supervised_stance_3_fts",
'sum_extension.supervised_stance.am':"sum_extension.supervised_stance_3_fts",
'sum_extension.supervised_stance.pm':"sum_extension.supervised_stance_3_fts",
'sum_favorite_count':'sum_favorite_count',
'sum_user.favourites_count':'sum_user.favourites_count',
'sum_user.followers_count':'sum_user.followers_count',
'sum_user.friends_count':'sum_user.friends_count',
'sum_user.listed_count':'sum_user.listed_count'
}

twitter_sum_fts = [
'sum_extension.botometer.cap.english',
'sum_extension.botometer.cap.universal',
'sum_extension.botometer.categories.content',
'sum_extension.botometer.categories.friend',
'sum_extension.botometer.categories.network',
'sum_extension.botometer.categories.sentiment',
'sum_extension.botometer.categories.temporal',
'sum_extension.botometer.display_scores.english',
'sum_extension.botometer.display_scores.friend',
'sum_extension.botometer.display_scores.network',
'sum_extension.botometer.display_scores.sentiment',
'sum_extension.botometer.display_scores.temporal',
'sum_extension.botometer.display_scores.universal',
'sum_extension.botometer.display_scores.user',
'sum_extension.botometer.scores.english',
'sum_extension.botometer.scores.universal',
'sum_extension.sentiment_scores.negative',
'sum_extension.sentiment_scores.neutral',
'sum_extension.sentiment_scores.positive',
'sum_extension.supervised_stance',
'sum_extension.supervised_stance.?',
'sum_extension.supervised_stance.am',
'sum_extension.supervised_stance.pm',
'sum_favorite_count',
'sum_user.favourites_count',
'sum_user.followers_count',
'sum_user.friends_count',
'sum_user.listed_count',
]

sum_user_friends_count_fts = ["sum_user.friends_count"]


sum_user_followers_count_fts=['sum_user.followers_count']
sum_user_friends_count_fts = ['sum_user.friends_count']
sum_user_listed_count_fts = ['sum_user.listed_count']

twitter_sum_botometer_cap_fts = [
'sum_extension.botometer.cap.english',
'sum_extension.botometer.cap.universal'
]

twitter_sum_botometer_cat_fts = [
'sum_extension.botometer.categories.content',
'sum_extension.botometer.categories.friend',
'sum_extension.botometer.categories.network',
'sum_extension.botometer.categories.sentiment',
'sum_extension.botometer.categories.temporal'
]

twitter_sum_botometer_display_score_fts = [
'sum_extension.botometer.display_scores.english',
'sum_extension.botometer.display_scores.friend',
'sum_extension.botometer.display_scores.network',
'sum_extension.botometer.display_scores.sentiment',
'sum_extension.botometer.display_scores.temporal',
'sum_extension.botometer.display_scores.universal',
'sum_extension.botometer.display_scores.user'
]

twitter_sum_botometer_score_fts =[
'sum_extension.botometer.scores.english',
'sum_extension.botometer.scores.universal',

]

twitter_sum_sentiment_score_fts =[
'sum_extension.sentiment_scores.negative',
'sum_extension.sentiment_scores.neutral',
'sum_extension.sentiment_scores.positive'

]

twitter_sum_reglar_stance_fts =['sum_extension.supervised_stance']

sum_favorite_count_fts =['sum_favorite_count']

sum_user_favourites_count_fts = ['sum_user.favourites_count']

twitter_sum_stance_3_fts = [

'sum_extension.supervised_stance.?',
'sum_extension.supervised_stance.am',
'sum_extension.supervised_stance.pm',
]

infoID_1hot_fts=[
"arrests",
"arrests/opposition",
"guaido/legitimate",
"international/aid",
"international/aid_rejected",
"international/respect_sovereignty",
"maduro/cuba_support",
"maduro/dictator",
"maduro/legitimate",
"maduro/narco",
"military",
"military/desertions",
"other/anti_socialism",
"other/censorship_outage",
"other/chavez",
"other/chavez/anti",
"protests",
"violence"
]

reddit_fts=[
"reddit_post",
"reddit_comment"
]

twitter_1hot_fts=[
"twitter_retweet",
"twitter_tweet"
]

youtube_1hot_fts=[
"youtube_comment",
"youtube_video"
]

actionType_1hot_fts =[
"twitter_retweet",
"twitter_tweet",
"twitter_quote",
"twitter_reply"
"youtube_comment",
"youtube_video"
]

gdelt_num_mentions=[
"NumMentions"
]

gdelt_sentiment_fts=[
"AvgTone",
"GoldsteinScale"
]

aux_twitter_action_fts=[
"aux_twitter_retweet",
"aux_twitter_tweet"
]

aux_youtube_action_fts=[
"aux_youtube_video",
"aux_youtube_comment"
]

aux_infoID_fts=[
"aux_arrests",
"aux_arrests/opposition",
"aux_guaido/legitimate",
"aux_international/aid",
"aux_international/aid_rejected",
"aux_international/respect_sovereignty",
"aux_maduro/cuba_support",
"aux_maduro/dictator",
"aux_maduro/legitimate",
"aux_maduro/narco",
"aux_military",
"aux_military/desertions",
"aux_other/anti_socialism",
"aux_other/censorship_outage",
"aux_other/chavez",
"aux_other/chavez/anti",
"aux_protests",
"aux_violence"

]

aux_youtube_user_age_fts=[
"aux_num_youtube_new_users",
"aux_num_youtube_old_users"
]

aux_twitter_user_age_fts=[
"aux_num_twitter_new_users",
"aux_num_twitter_old_users"
]

target_action_value_fts = ["target_action_value"]

ft_category_to_list_dict={
'target_action_value':target_action_value_fts,

'youtube_1hot_fts':youtube_1hot_fts,
'twitter_1hot_fts':twitter_1hot_fts,
'infoID_1hot_fts':infoID_1hot_fts,

'reddit_fts':reddit_fts,
'gdelt_num_mentions':gdelt_num_mentions,
'gdelt_sentiment_fts':gdelt_sentiment_fts,

'aux_twitter_user_age_fts':aux_twitter_user_age_fts,
'aux_youtube_user_age_fts':aux_youtube_user_age_fts,

'aux_youtube_action_fts':aux_youtube_action_fts,
'aux_twitter_action_fts':aux_twitter_action_fts,

'aux_infoID_fts':aux_infoID_fts,

'sum_extension.supervised_stance_single_ft':twitter_sum_reglar_stance_fts ,
'sum_user.friends_count':sum_user_friends_count_fts,
'sum_extension.botometer.scores':twitter_sum_botometer_score_fts,
'sum_extension.botometer.display_scores':twitter_sum_botometer_display_score_fts,
'sum_user.favourites_count':['sum_user.favourites_count'],
'sum_extension.sentiment_scores':twitter_sum_sentiment_score_fts,
'sum_user.followers_count':['sum_user.followers_count'],
'sum_user.listed_count':['sum_user.listed_count'],
'sum_extension.botometer.cap':twitter_sum_botometer_cap_fts,
'sum_extension.supervised_stance_3_fts':twitter_sum_stance_3_fts,
'sum_favorite_count':['sum_favorite_count'],
'sum_extension.botometer.categories':twitter_sum_botometer_cat_fts
}

new_twitter_ft_cols=[
'max_extension.botometer.cap.english',
'max_extension.botometer.cap.universal',
'max_extension.botometer.categories.content',
'max_extension.botometer.categories.friend',
'max_extension.botometer.categories.network',
'max_extension.botometer.categories.sentiment',
'max_extension.botometer.categories.temporal',
'max_extension.botometer.display_scores.english',
'max_extension.botometer.display_scores.friend',
'max_extension.botometer.display_scores.network',
'max_extension.botometer.display_scores.sentiment',
'max_extension.botometer.display_scores.temporal',
'max_extension.botometer.display_scores.universal',
'max_extension.botometer.display_scores.user',
'max_extension.botometer.scores.english',
'max_extension.botometer.scores.universal',
'max_extension.sentiment_scores.negative',
'max_extension.sentiment_scores.neutral',
'max_extension.sentiment_scores.positive',
'max_extension.supervised_stance',
'max_extension.supervised_stance.?',
'max_extension.supervised_stance.am',
'max_extension.supervised_stance.pm',
'max_favorite_count',
'max_user.favourites_count',
'max_user.followers_count',
'max_user.friends_count',
'max_user.listed_count',
'mean_extension.botometer.cap.english',
'mean_extension.botometer.cap.universal',
'mean_extension.botometer.categories.content',
'mean_extension.botometer.categories.friend',
'mean_extension.botometer.categories.network',
'mean_extension.botometer.categories.sentiment',
'mean_extension.botometer.categories.temporal',
'mean_extension.botometer.display_scores.english',
'mean_extension.botometer.display_scores.friend',
'mean_extension.botometer.display_scores.network',
'mean_extension.botometer.display_scores.sentiment',
'mean_extension.botometer.display_scores.temporal',
'mean_extension.botometer.display_scores.universal',
'mean_extension.botometer.display_scores.user',
'mean_extension.botometer.scores.english',
'mean_extension.botometer.scores.universal',
'mean_extension.sentiment_scores.negative',
'mean_extension.sentiment_scores.neutral',
'mean_extension.sentiment_scores.positive',
'mean_extension.supervised_stance',
'mean_extension.supervised_stance.?',
'mean_extension.supervised_stance.am',
'mean_extension.supervised_stance.pm',
'mean_favorite_count',
'mean_user.favourites_count',
'mean_user.followers_count',
'mean_user.friends_count',
'mean_user.listed_count',
'median_extension.botometer.cap.english',
'median_extension.botometer.cap.universal',
'median_extension.botometer.categories.content',
'median_extension.botometer.categories.friend',
'median_extension.botometer.categories.network',
'median_extension.botometer.categories.sentiment',
'median_extension.botometer.categories.temporal',
'median_extension.botometer.display_scores.english',
'median_extension.botometer.display_scores.friend',
'median_extension.botometer.display_scores.network',
'median_extension.botometer.display_scores.sentiment',
'median_extension.botometer.display_scores.temporal',
'median_extension.botometer.display_scores.universal',
'median_extension.botometer.display_scores.user',
'median_extension.botometer.scores.english',
'median_extension.botometer.scores.universal',
'median_extension.sentiment_scores.negative',
'median_extension.sentiment_scores.neutral',
'median_extension.sentiment_scores.positive',
'median_extension.supervised_stance',
'median_extension.supervised_stance.?',
'median_extension.supervised_stance.am',
'median_extension.supervised_stance.pm',
'median_favorite_count',
'median_user.favourites_count',
'median_user.followers_count',
'median_user.friends_count',
'median_user.listed_count',
'min_extension.botometer.cap.english',
'min_extension.botometer.cap.universal',
'min_extension.botometer.categories.content',
'min_extension.botometer.categories.friend',
'min_extension.botometer.categories.network',
'min_extension.botometer.categories.sentiment',
'min_extension.botometer.categories.temporal',
'min_extension.botometer.display_scores.english',
'min_extension.botometer.display_scores.friend',
'min_extension.botometer.display_scores.network',
'min_extension.botometer.display_scores.sentiment',
'min_extension.botometer.display_scores.temporal',
'min_extension.botometer.display_scores.universal',
'min_extension.botometer.display_scores.user',
'min_extension.botometer.scores.english',
'min_extension.botometer.scores.universal',
'min_extension.sentiment_scores.negative',
'min_extension.sentiment_scores.neutral',
'min_extension.sentiment_scores.positive',
'min_extension.supervised_stance',
'min_extension.supervised_stance.?',
'min_extension.supervised_stance.am',
'min_extension.supervised_stance.pm',
'min_favorite_count',
'min_user.favourites_count',
'min_user.followers_count',
'min_user.friends_count',
'min_user.listed_count',
'std_extension.botometer.cap.english',
'std_extension.botometer.cap.universal',
'std_extension.botometer.categories.content',
'std_extension.botometer.categories.friend',
'std_extension.botometer.categories.network',
'std_extension.botometer.categories.sentiment',
'std_extension.botometer.categories.temporal',
'std_extension.botometer.display_scores.english',
'std_extension.botometer.display_scores.friend',
'std_extension.botometer.display_scores.network',
'std_extension.botometer.display_scores.sentiment',
'std_extension.botometer.display_scores.temporal',
'std_extension.botometer.display_scores.universal',
'std_extension.botometer.display_scores.user',
'std_extension.botometer.scores.english',
'std_extension.botometer.scores.universal',
'std_extension.sentiment_scores.negative',
'std_extension.sentiment_scores.neutral',
'std_extension.sentiment_scores.positive',
'std_extension.supervised_stance',
'std_extension.supervised_stance.?',
'std_extension.supervised_stance.am',
'std_extension.supervised_stance.pm',
'std_favorite_count',
'std_user.favourites_count',
'std_user.followers_count',
'std_user.friends_count',
'std_user.listed_count',
'sum_extension.botometer.cap.english',
'sum_extension.botometer.cap.universal',
'sum_extension.botometer.categories.content',
'sum_extension.botometer.categories.friend',
'sum_extension.botometer.categories.network',
'sum_extension.botometer.categories.sentiment',
'sum_extension.botometer.categories.temporal',
'sum_extension.botometer.display_scores.english',
'sum_extension.botometer.display_scores.friend',
'sum_extension.botometer.display_scores.network',
'sum_extension.botometer.display_scores.sentiment',
'sum_extension.botometer.display_scores.temporal',
'sum_extension.botometer.display_scores.universal',
'sum_extension.botometer.display_scores.user',
'sum_extension.botometer.scores.english',
'sum_extension.botometer.scores.universal',
'sum_extension.sentiment_scores.negative',
'sum_extension.sentiment_scores.neutral',
'sum_extension.sentiment_scores.positive',
'sum_extension.supervised_stance',
'sum_extension.supervised_stance.?',
'sum_extension.supervised_stance.am',
'sum_extension.supervised_stance.pm',
'sum_favorite_count',
'sum_user.favourites_count',
'sum_user.followers_count',
'sum_user.friends_count',
'sum_user.listed_count',
]

