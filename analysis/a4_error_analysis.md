# A4: Error analysis of the joint decision model (Romanized vs Bengali)

Same 1999 utterances in English, Bengali (Bangla script) and Romanized form; differences come from the written form alone.

## Outcome on parallel items (Bengali vs Romanized)

| Outcome | Items | Share |
|---|---|---|
| Correct in both | 1254 | 62.7% |
| Correct only in Bangla script | 330 | 16.5% |
| Correct only in Romanized | 83 | 4.2% |
| Wrong in both | 332 | 16.6% |

## Accuracy by domain (largest Bengali-minus-Romanized gap first)

| Domain | Items | English | Bengali | Romanized | Gap (BN - ROM, points) |
|---|---|---|---|---|---|
| weather | 110 | 1.000 | 0.909 | 0.591 | +31.8 |
| news | 97 | 0.907 | 0.784 | 0.495 | +28.9 |
| recommendation | 68 | 0.882 | 0.853 | 0.618 | +23.5 |
| qa | 216 | 0.819 | 0.741 | 0.556 | +18.5 |
| datetime | 64 | 0.938 | 0.891 | 0.734 | +15.6 |
| alarm | 59 | 0.983 | 0.983 | 0.831 | +15.3 |
| calendar | 332 | 0.870 | 0.834 | 0.714 | +12.0 |
| general | 126 | 0.437 | 0.333 | 0.214 | +11.9 |
| music | 44 | 0.795 | 0.773 | 0.659 | +11.4 |
| social | 73 | 0.904 | 0.904 | 0.795 | +11.0 |
| audio | 30 | 0.767 | 0.700 | 0.600 | +10.0 |
| lists | 89 | 0.899 | 0.854 | 0.775 | +7.9 |
| email | 176 | 0.926 | 0.886 | 0.812 | +7.4 |
| iot | 88 | 0.886 | 0.795 | 0.727 | +6.8 |
| transport | 90 | 0.900 | 0.856 | 0.789 | +6.7 |
| play | 259 | 0.822 | 0.776 | 0.734 | +4.2 |
| cooking | 40 | 0.900 | 0.725 | 0.750 | -2.5 |
| takeaway | 38 | 0.842 | 0.684 | 0.789 | -10.5 |

## Most frequent confusions, Bengali (gold -> predicted)

415 errors; 72% cross into a different domain.

| Gold intent | Predicted intent | Count |
|---|---|---|
| general_quirky | qa_factoid | 20 |
| calendar_set | calendar_query | 10 |
| calendar_query | calendar_set | 9 |
| calendar_query | recommendation_events | 9 |
| play_radio | play_music | 7 |
| play_music | music_likeness | 6 |
| qa_factoid | datetime_query | 6 |
| news_query | qa_factoid | 5 |

## Most frequent confusions, Romanized (gold -> predicted)

662 errors; 78% cross into a different domain.

| Gold intent | Predicted intent | Count |
|---|---|---|
| calendar_query | recommendation_events | 18 |
| news_query | qa_factoid | 12 |
| general_quirky | qa_factoid | 12 |
| qa_factoid | cooking_recipe | 11 |
| calendar_set | calendar_query | 9 |
| weather_query | recommendation_events | 8 |
| general_quirky | cooking_recipe | 8 |
| play_music | music_likeness | 7 |

## Examples: correct in Bangla script, wrong in Romanized (most confident errors)

| Gold | Bangla script | Romanized | Romanized prediction (confidence) |
|---|---|---|---|
| email_sendemail | আমি ইমেলটি আমার নতুন ইমেইল ঠিকানা a. b. c. at gmail dot com এ প্রেরণ করতে চাই | ami email ti amar notun email thikana a. b. c. at gmail dot com e preron korte chai | email_addcontact (1.00) |
| lists_query | আমার প্লেলিস্টে কি আছে | amar playlist e ki ache | music_query (0.99) |
| datetime_convert | সময় অঞ্চল রূপান্তর করুন | somoy onchol rupantor korun | qa_currency (0.99) |
| music_query | এই মুহূর্তে আমার প্রিয় সঙ্গীত কি | ei muhurte amar priyo songit ki | music_likeness (0.99) |
| news_query | প্রথম আলো পোস্ট ওয়েবসাইটে যান | prothom alo post oyebsaite jan | social_post (0.99) |
| news_query | প্রথম আলো আবহাওয়া সম্পর্কিত কোনো নিবন্ধ আমাকে দেখান | prothom alo abohawa somporkit kono nibondho amake dekhan | weather_query (0.99) |
| qa_definition | সিরি একটি রংধনু বর্ণনা করে | siri ekti rongdhnu bornona kore | iot_hue_lightchange (0.99) |
| play_music | আমি লেডি গাগা পোকার ফেস শুনতে চাই | ami ledi gaga poker fes shunte chai | play_game (0.98) |
