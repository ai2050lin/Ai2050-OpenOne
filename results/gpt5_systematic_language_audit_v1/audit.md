# Systematic Language Audit v1

## Model Overview

| model | complete | n | full acc | mean acc | first acc | first/full disagree |
|---|---:|---:|---:|---:|---:|---:|
| deepseek7b | True | 900 | 71.11% | 70.56% | 71.11% | 0 |
| glm4 | True | 900 | 78.89% | 78.33% | 78.89% | 0 |
| qwen3 | True | 900 | 95.56% | 95.00% | 95.56% | 0 |

## Category Accuracy

| model | category | n | unique | dup factor | full acc | failures | low-margin correct |
|---|---|---:|---:|---:|---:|---:|---:|
| deepseek7b | comparison | 100 | 20 | 5.0 | 75.00% | 25 | 5 |
| deepseek7b | conditional | 100 | 20 | 5.0 | 95.00% | 5 | 5 |
| deepseek7b | negation_yesno | 100 | 20 | 5.0 | 50.00% | 50 | 0 |
| deepseek7b | passive_agent | 100 | 20 | 5.0 | 75.00% | 25 | 15 |
| deepseek7b | quantifier | 100 | 10 | 10.0 | 40.00% | 60 | 0 |
| deepseek7b | recursive_binding | 100 | 20 | 5.0 | 65.00% | 35 | 20 |
| deepseek7b | svo_agent | 100 | 20 | 5.0 | 100.00% | 0 | 0 |
| deepseek7b | temporal | 100 | 10 | 10.0 | 40.00% | 60 | 20 |
| deepseek7b | translation | 100 | 50 | 2.0 | 100.00% | 0 | 0 |
| glm4 | comparison | 100 | 20 | 5.0 | 90.00% | 10 | 5 |
| glm4 | conditional | 100 | 20 | 5.0 | 100.00% | 0 | 5 |
| glm4 | negation_yesno | 100 | 20 | 5.0 | 50.00% | 50 | 0 |
| glm4 | passive_agent | 100 | 20 | 5.0 | 90.00% | 10 | 15 |
| glm4 | quantifier | 100 | 10 | 10.0 | 50.00% | 50 | 10 |
| glm4 | recursive_binding | 100 | 20 | 5.0 | 80.00% | 20 | 25 |
| glm4 | svo_agent | 100 | 20 | 5.0 | 100.00% | 0 | 0 |
| glm4 | temporal | 100 | 10 | 10.0 | 50.00% | 50 | 10 |
| glm4 | translation | 100 | 50 | 2.0 | 100.00% | 0 | 0 |
| qwen3 | comparison | 100 | 20 | 5.0 | 100.00% | 0 | 10 |
| qwen3 | conditional | 100 | 20 | 5.0 | 95.00% | 5 | 0 |
| qwen3 | negation_yesno | 100 | 20 | 5.0 | 100.00% | 0 | 10 |
| qwen3 | passive_agent | 100 | 20 | 5.0 | 100.00% | 0 | 0 |
| qwen3 | quantifier | 100 | 10 | 10.0 | 100.00% | 0 | 10 |
| qwen3 | recursive_binding | 100 | 20 | 5.0 | 85.00% | 15 | 35 |
| qwen3 | svo_agent | 100 | 20 | 5.0 | 100.00% | 0 | 0 |
| qwen3 | temporal | 100 | 10 | 10.0 | 80.00% | 20 | 10 |
| qwen3 | translation | 100 | 50 | 2.0 | 100.00% | 0 | 0 |

## Cross Model Overlap

| category | n | all correct | all wrong | mixed | wrong by model |
|---|---:|---:|---:|---:|---|
| comparison | 100 | 70 | 0 | 30 | deepseek7b:25, glm4:10 |
| conditional | 100 | 95 | 0 | 5 | deepseek7b:5, qwen3:5 |
| negation_yesno | 100 | 50 | 0 | 50 | deepseek7b:50, glm4:50 |
| passive_agent | 100 | 75 | 0 | 25 | deepseek7b:25, glm4:10 |
| quantifier | 100 | 40 | 0 | 60 | deepseek7b:60, glm4:50 |
| recursive_binding | 100 | 50 | 5 | 45 | deepseek7b:35, glm4:20, qwen3:15 |
| svo_agent | 100 | 100 | 0 | 0 |  |
| temporal | 100 | 30 | 10 | 60 | deepseek7b:60, glm4:50, qwen3:20 |
| translation | 100 | 100 | 0 | 0 |  |

## Worst Failures

### deepseek7b

#### comparison
- `comparison_005` margin=-6.250; answer=' tea'; predicted=' ice'; prompt='tea is hotter than ice. The hotter thing is'
- `comparison_025` margin=-6.250; answer=' tea'; predicted=' ice'; prompt='tea is hotter than ice. The hotter thing is'
- `comparison_045` margin=-6.250; answer=' tea'; predicted=' ice'; prompt='tea is hotter than ice. The hotter thing is'
- `comparison_065` margin=-6.250; answer=' tea'; predicted=' ice'; prompt='tea is hotter than ice. The hotter thing is'
- `comparison_085` margin=-6.250; answer=' tea'; predicted=' ice'; prompt='tea is hotter than ice. The hotter thing is'

#### conditional
- `conditional_001` margin=-0.312; answer=' up'; predicted=' asleep'; prompt='If the alarm rings, the guard wakes up. The alarm rings. The result is'
- `conditional_021` margin=-0.312; answer=' up'; predicted=' asleep'; prompt='If the alarm rings, the guard wakes up. The alarm rings. The result is'
- `conditional_041` margin=-0.312; answer=' up'; predicted=' asleep'; prompt='If the alarm rings, the guard wakes up. The alarm rings. The result is'
- `conditional_061` margin=-0.312; answer=' up'; predicted=' asleep'; prompt='If the alarm rings, the guard wakes up. The alarm rings. The result is'
- `conditional_081` margin=-0.312; answer=' up'; predicted=' asleep'; prompt='If the alarm rings, the guard wakes up. The alarm rings. The result is'

#### negation_yesno
- `negation_yesno_000` margin=-3.188; answer=' no'; predicted=' yes'; prompt='The door is not open. Is the door open? Answer yes or no:'
- `negation_yesno_020` margin=-3.188; answer=' no'; predicted=' yes'; prompt='The door is not open. Is the door open? Answer yes or no:'
- `negation_yesno_040` margin=-3.188; answer=' no'; predicted=' yes'; prompt='The door is not open. Is the door open? Answer yes or no:'
- `negation_yesno_060` margin=-3.188; answer=' no'; predicted=' yes'; prompt='The door is not open. Is the door open? Answer yes or no:'
- `negation_yesno_016` margin=-2.375; answer=' no'; predicted=' yes'; prompt='The knife is not open. Is the knife open? Answer yes or no:'

#### passive_agent
- `passive_agent_010` margin=-6.250; answer=' nurse'; predicted=' school'; prompt='In the sentence "the school is followed by the nurse", the doer is the'
- `passive_agent_030` margin=-6.250; answer=' nurse'; predicted=' school'; prompt='In the sentence "the school is followed by the nurse", the doer is the'
- `passive_agent_050` margin=-6.250; answer=' nurse'; predicted=' school'; prompt='In the sentence "the school is followed by the nurse", the doer is the'
- `passive_agent_070` margin=-6.250; answer=' nurse'; predicted=' school'; prompt='In the sentence "the school is followed by the nurse", the doer is the'
- `passive_agent_090` margin=-6.250; answer=' nurse'; predicted=' school'; prompt='In the sentence "the school is followed by the nurse", the doer is the'

#### quantifier
- `quantifier_004` margin=-2.562; answer=' no'; predicted=' yes'; prompt='Few keys came to the room. Did many keys come? Answer yes or no:'
- `quantifier_014` margin=-2.562; answer=' no'; predicted=' yes'; prompt='Few keys came to the room. Did many keys come? Answer yes or no:'
- `quantifier_024` margin=-2.562; answer=' no'; predicted=' yes'; prompt='Few keys came to the room. Did many keys come? Answer yes or no:'
- `quantifier_034` margin=-2.562; answer=' no'; predicted=' yes'; prompt='Few keys came to the room. Did many keys come? Answer yes or no:'
- `quantifier_044` margin=-2.562; answer=' no'; predicted=' yes'; prompt='Few keys came to the room. Did many keys come? Answer yes or no:'

#### recursive_binding
- `recursive_binding_007` margin=-3.125; answer=' nurse'; predicted=' queen'; prompt='The nurse that the queen followed was clean. The clean one was the'
- `recursive_binding_027` margin=-3.125; answer=' nurse'; predicted=' queen'; prompt='The nurse that the queen followed was clean. The clean one was the'
- `recursive_binding_047` margin=-3.125; answer=' nurse'; predicted=' queen'; prompt='The nurse that the queen followed was clean. The clean one was the'
- `recursive_binding_067` margin=-3.125; answer=' nurse'; predicted=' queen'; prompt='The nurse that the queen followed was clean. The clean one was the'
- `recursive_binding_087` margin=-3.125; answer=' nurse'; predicted=' queen'; prompt='The nurse that the queen followed was clean. The clean one was the'

#### temporal
- `temporal_002` margin=-1.812; answer=' present'; predicted=' past'; prompt='Now, Sam washed his hands. The washing is happening in the'
- `temporal_012` margin=-1.812; answer=' present'; predicted=' past'; prompt='Now, Sam washed his hands. The washing is happening in the'
- `temporal_022` margin=-1.812; answer=' present'; predicted=' past'; prompt='Now, Sam washed his hands. The washing is happening in the'
- `temporal_032` margin=-1.812; answer=' present'; predicted=' past'; prompt='Now, Sam washed his hands. The washing is happening in the'
- `temporal_042` margin=-1.812; answer=' present'; predicted=' past'; prompt='Now, Sam washed his hands. The washing is happening in the'

### glm4

#### comparison
- `comparison_005` margin=-0.625; answer=' tea'; predicted=' ice'; prompt='tea is hotter than ice. The hotter thing is'
- `comparison_025` margin=-0.625; answer=' tea'; predicted=' ice'; prompt='tea is hotter than ice. The hotter thing is'
- `comparison_045` margin=-0.625; answer=' tea'; predicted=' ice'; prompt='tea is hotter than ice. The hotter thing is'
- `comparison_065` margin=-0.625; answer=' tea'; predicted=' ice'; prompt='tea is hotter than ice. The hotter thing is'
- `comparison_085` margin=-0.625; answer=' tea'; predicted=' ice'; prompt='tea is hotter than ice. The hotter thing is'

#### negation_yesno
- `negation_yesno_004` margin=-2.250; answer=' no'; predicted=' yes'; prompt='The place is not open. Is the place open? Answer yes or no:'
- `negation_yesno_024` margin=-2.250; answer=' no'; predicted=' yes'; prompt='The place is not open. Is the place open? Answer yes or no:'
- `negation_yesno_044` margin=-2.250; answer=' no'; predicted=' yes'; prompt='The place is not open. Is the place open? Answer yes or no:'
- `negation_yesno_064` margin=-2.250; answer=' no'; predicted=' yes'; prompt='The place is not open. Is the place open? Answer yes or no:'
- `negation_yesno_084` margin=-2.250; answer=' no'; predicted=' yes'; prompt='The place is not open. Is the place open? Answer yes or no:'

#### passive_agent
- `passive_agent_010` margin=-1.875; answer=' nurse'; predicted=' school'; prompt='In the sentence "the school is followed by the nurse", the doer is the'
- `passive_agent_030` margin=-1.875; answer=' nurse'; predicted=' school'; prompt='In the sentence "the school is followed by the nurse", the doer is the'
- `passive_agent_050` margin=-1.875; answer=' nurse'; predicted=' school'; prompt='In the sentence "the school is followed by the nurse", the doer is the'
- `passive_agent_070` margin=-1.875; answer=' nurse'; predicted=' school'; prompt='In the sentence "the school is followed by the nurse", the doer is the'
- `passive_agent_090` margin=-1.875; answer=' nurse'; predicted=' school'; prompt='In the sentence "the school is followed by the nurse", the doer is the'

#### quantifier
- `quantifier_004` margin=-2.188; answer=' no'; predicted=' yes'; prompt='Few keys came to the room. Did many keys come? Answer yes or no:'
- `quantifier_014` margin=-2.188; answer=' no'; predicted=' yes'; prompt='Few keys came to the room. Did many keys come? Answer yes or no:'
- `quantifier_024` margin=-2.188; answer=' no'; predicted=' yes'; prompt='Few keys came to the room. Did many keys come? Answer yes or no:'
- `quantifier_034` margin=-2.188; answer=' no'; predicted=' yes'; prompt='Few keys came to the room. Did many keys come? Answer yes or no:'
- `quantifier_044` margin=-2.188; answer=' no'; predicted=' yes'; prompt='Few keys came to the room. Did many keys come? Answer yes or no:'

#### recursive_binding
- `recursive_binding_007` margin=-1.625; answer=' nurse'; predicted=' queen'; prompt='The nurse that the queen followed was clean. The clean one was the'
- `recursive_binding_027` margin=-1.625; answer=' nurse'; predicted=' queen'; prompt='The nurse that the queen followed was clean. The clean one was the'
- `recursive_binding_047` margin=-1.625; answer=' nurse'; predicted=' queen'; prompt='The nurse that the queen followed was clean. The clean one was the'
- `recursive_binding_067` margin=-1.625; answer=' nurse'; predicted=' queen'; prompt='The nurse that the queen followed was clean. The clean one was the'
- `recursive_binding_087` margin=-1.625; answer=' nurse'; predicted=' queen'; prompt='The nurse that the queen followed was clean. The clean one was the'

#### temporal
- `temporal_006` margin=-1.250; answer=' future'; predicted=' past'; prompt='Tomorrow, the train leaves the station. The leaving happens in the'
- `temporal_016` margin=-1.250; answer=' future'; predicted=' past'; prompt='Tomorrow, the train leaves the station. The leaving happens in the'
- `temporal_026` margin=-1.250; answer=' future'; predicted=' past'; prompt='Tomorrow, the train leaves the station. The leaving happens in the'
- `temporal_036` margin=-1.250; answer=' future'; predicted=' past'; prompt='Tomorrow, the train leaves the station. The leaving happens in the'
- `temporal_046` margin=-1.250; answer=' future'; predicted=' past'; prompt='Tomorrow, the train leaves the station. The leaving happens in the'

### qwen3

#### conditional
- `conditional_001` margin=-2.625; answer=' up'; predicted=' asleep'; prompt='If the alarm rings, the guard wakes up. The alarm rings. The result is'
- `conditional_021` margin=-2.625; answer=' up'; predicted=' asleep'; prompt='If the alarm rings, the guard wakes up. The alarm rings. The result is'
- `conditional_041` margin=-2.625; answer=' up'; predicted=' asleep'; prompt='If the alarm rings, the guard wakes up. The alarm rings. The result is'
- `conditional_061` margin=-2.625; answer=' up'; predicted=' asleep'; prompt='If the alarm rings, the guard wakes up. The alarm rings. The result is'
- `conditional_081` margin=-2.625; answer=' up'; predicted=' asleep'; prompt='If the alarm rings, the guard wakes up. The alarm rings. The result is'

#### recursive_binding
- `recursive_binding_074` margin=-0.755; answer=' farmer'; predicted=' dog'; prompt='The farmer that helped the dog was green. The green one was the'
- `recursive_binding_014` margin=-0.750; answer=' farmer'; predicted=' dog'; prompt='The farmer that helped the dog was green. The green one was the'
- `recursive_binding_034` margin=-0.750; answer=' farmer'; predicted=' dog'; prompt='The farmer that helped the dog was green. The green one was the'
- `recursive_binding_054` margin=-0.750; answer=' farmer'; predicted=' dog'; prompt='The farmer that helped the dog was green. The green one was the'
- `recursive_binding_094` margin=-0.750; answer=' farmer'; predicted=' dog'; prompt='The farmer that helped the dog was green. The green one was the'

#### temporal
- `temporal_003` margin=-1.438; answer=' before'; predicted=' after'; prompt='Before dinner, the guard opened the gate. The opening happened'
- `temporal_013` margin=-1.438; answer=' before'; predicted=' after'; prompt='Before dinner, the guard opened the gate. The opening happened'
- `temporal_023` margin=-1.438; answer=' before'; predicted=' after'; prompt='Before dinner, the guard opened the gate. The opening happened'
- `temporal_033` margin=-1.438; answer=' before'; predicted=' after'; prompt='Before dinner, the guard opened the gate. The opening happened'
- `temporal_043` margin=-1.438; answer=' before'; predicted=' after'; prompt='Before dinner, the guard opened the gate. The opening happened'
