import re
import json
from utils import recursive_word_chunker
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "maya-research/maya1", 
    torch_dtype=torch.bfloat16, 
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    "maya-research/maya1",
    trust_remote_code=True
)

CODE_START_TOKEN_ID = 128257
CODE_END_TOKEN_ID = 128258
CODE_TOKEN_OFFSET = 128266
SNAC_MIN_ID = 128266
SNAC_MAX_ID = 156937
SNAC_TOKENS_PER_FRAME = 7

SOH_ID = 128259
EOH_ID = 128260
SOA_ID = 128261
BOS_ID = 128000
TEXT_EOT_ID = 128009


def build_prompt(tokenizer, description: str, text: str) -> str:
    """Build formatted prompt for Maya1."""
    soh_token = tokenizer.decode([SOH_ID])
    eoh_token = tokenizer.decode([EOH_ID])
    soa_token = tokenizer.decode([SOA_ID])
    sos_token = tokenizer.decode([CODE_START_TOKEN_ID])
    eot_token = tokenizer.decode([TEXT_EOT_ID])
    bos_token = tokenizer.bos_token
    
    formatted_text = f'<description="{description}"> {text}'
    
    prompt = (
        soh_token + bos_token + formatted_text + eot_token +
        eoh_token + soa_token + sos_token
    )
    
    return prompt

raw_samples = [
"""Hello, welcome to this guided meditation session, I'm here to help you relax and focus before your job interview tomorrow.

Find a quiet and comfortable place to sit, close your eyes and take a deep breath in through your nose, and exhale out through your mouth. [PAUSE:2.0] Feel the air fill your lungs and then release any tension as you breathe out.

Imagine yourself standing in a peaceful forest, surrounded by tall trees that stretch up towards the sky. The sun shines gently through the leaves, casting dappled shadows on the ground below. [PAUSE:4.0] With each breath, feel your body relax and let go of any anxiety or worries.

Now, bring your attention to your feet, feeling the weight of your body distributed evenly on the ground. [PAUSE:1.0] As you breathe in, imagine fresh energy and calmness flowing into your feet, and as you breathe out, imagine any tension or anxiety leaving your body through the soles of your feet.

Imagine a warm, soothing light beginning to fill your body, starting at the crown of your head and flowing down through your face, neck, and shoulders. [PAUSE:3.0] As this light reaches your heart, it fills you with confidence and self-assurance, reminding you of your skills and qualifications.

With each breath, repeat a gentle affirmation to yourself: "I am capable and prepared for this interview." [PAUSE:2.0] Allow these words to sink deeply into your mind, filling you with a sense of calm and focus.

Imagine yourself acing the interview, answering questions with ease and confidence. [PAUSE:2.0] Visualize the interviewer smiling and nodding in approval, feeling impressed by your skills and experience.

As you continue to breathe deeply, bring your attention to the present moment, focusing on your breath and the sensation of your body relaxing. [PAUSE:3.0] Remember, you've prepared well for this interview, and you're ready to shine.

Take one final, deep breath in, and as you exhale, imagine any remaining anxiety or worry leaving your body. [PAUSE:4.0] When you're ready, slowly open your eyes, and remember to take this sense of calm and focus with you into your interview tomorrow.""",

"""Hello, take a deep breath in and allow yourself to settle into this moment. 

You're about to embark on an incredible journey, one that will bring you closer to your wildest dreams. [PAUSE:1.0] 

As you sit here, feel the weight of the chair beneath you, the sensation of the air on your skin, and the gentle cadence of your breath. [PAUSE:2.0] 

Imagine yourself walking into that interview room, feeling confident, poised, and ready to take on the world. [PAUSE:3.0] 

Visualise the interviewer smiling, nodding, and impressed by your answers. [PAUSE:2.0] 

Notice how your body feels when you're experiencing excitement and anticipation. [PAUSE:1.0] 

Take a deep breath in for four, hold it for seven, and exhale all your tension for eight. [PAUSE:4.0][PAUSE:7.0][PAUSE:8.0] 

As you inhale, repeat the phrase 'I am calm and capable' to yourself. [PAUSE:2.0] 

Allow the words to sink deep into your mind and heart. [PAUSE:3.0] 

Now, bring your attention to your feet. Feel the ground beneath you, the sensation of your toes touching the floor. [PAUSE:1.0] 

As you breathe out, imagine any anxiety or self-doubt leaving your body. [PAUSE:3.0] 

Repeat this phrase 'I am enough' to yourself, feeling the truth of these words resonating deep within your being. [PAUSE:2.0] 

Imagine yourself acing that interview, feeling proud and accomplished. [PAUSE:3.0] 

As you sit here, take one final, deep breath in, hold it for a moment, and exhale, feeling refreshed, renewed, and ready to take on the world.""",

"""Hello, welcome to this guided meditation session. I'm here to help you relax and unwind, preparing your body and mind for a restful night's sleep. Find a quiet, comfortable place to sit or lie down, with your back supported.

Close your eyes, and take a deep breath in through your nose. [PAUSE:4.0] Slowly exhale through your mouth, feeling any tension or stress leave your body. [PAUSE:6.0] Allow your eyelids to grow heavy, and let your gaze soften into a calm, inward focus.

Begin to breathe diaphragmatically, feeling your belly rise and fall with each breath. Imagine fresh air filling your lungs, nourishing your body and calming your mind. [PAUSE:2.0] As you inhale, say to yourself, "I am relaxed, I am calm." [PAUSE:1.0] Hold that thought for a moment.

Now, imagine yourself standing on a peaceful beach at sunset. Feel the warm sand beneath your feet, and the gentle ocean breeze on your skin. [PAUSE:5.0] Visualise the sky, a soft pink and orange hue, with the sun slowly dipping below the horizon.

As the sun disappears, the stars begin to twinkle in the night sky. Allow your gaze to rise upwards, feeling the vastness of the universe above you. [PAUSE:4.0] With each breath, feel any remaining tension or stress melting away, like the ebbing tide.

Imagine roots growing from the base of your spine, deep into the earth, anchoring you in stability and calm. [PAUSE:3.0] With each breath, feel yourself sinking deeper into relaxation, letting go of the day's worries and concerns.

As you continue to breathe deeply, allow your body to relax further, starting from the crown of your head, down to your shoulders, arms, chest, abdomen, lower back, hips, legs, and finally, your toes. [PAUSE:2.0] As each part of your body releases tension, feel a sense of relief and calm wash over you.

Now, imagine yourself in a peaceful, dreamless sleep, recharging and renewing your body and mind. [PAUSE:5.0] With each breath, feel yourself becoming heavier, more relaxed, and more at ease.

As you drift off to sleep, remember that you can return to this peaceful state whenever you need it. Take one final, deep breath in, and as you exhale, allow yourself to let go, surrendering to the restful sleep that awaits you.""",
"""Hello, welcome to this guided meditation session to help you overcome difficulty falling asleep.

Take a deep breath in through your nose and exhale out through your mouth, feeling any tension in your body slowly release. [PAUSE:4.0] 

Imagine yourself standing at the edge of a peaceful beach at sunset, the sky is painted with warm hues of orange and pink. [PAUSE:2.0] Listen to the sound of the waves gently lapping against the shore, feeling the rhythm of the ocean calming your mind. [PAUSE:2.0] 

Now, breathe in deeply, feeling the salty air fill your lungs, and exhale slowly, imagining any worries or stress leaving your body with each outbreath. [PAUSE:2.0] 

Visualise a warm, soothing light beginning to spread throughout your body, starting at the crown of your head and flowing down to your toes, relaxing your muscles and calming your mind. [PAUSE:2.0] 

As you inhale, repeat to yourself, 'I am relaxed, I am calm, I am at peace.' [PAUSE:2.0] 

Now, imagine yourself walking along the beach, feeling the soft sand beneath your feet, the cool breeze on your skin, and the sound of the waves in the distance. [PAUSE:2.0] 

As you breathe in, notice the sensation of your feet touching the ground, and as you exhale, feel any tension or stress melting away. [PAUSE:2.0] 

Continue to breathe deeply and imagine yourself becoming heavier, feeling your body sinking into the sand, becoming relaxed and calm. [PAUSE:2.0] 

Remember, you are safe, and you are in control. [PAUSE:2.0] 

Imagine yourself drifting off to sleep, feeling your body becoming heavier, your mind growing quieter, and your spirit becoming more peaceful. [PAUSE:2.0] 

As you exhale, repeat to yourself, 'I am relaxed, I am calm, I am at peace.' [PAUSE:2.0] 

Drifting deeper into relaxation, feeling your body and mind becoming more calm and peaceful with each breath. [PAUSE:2.0] 

And now, allow yourself to let go, to surrender to the present moment, and to drift off to sleep, feeling refreshed and renewed.""",
"""Hello, take a deep breath in and let's begin this journey to relaxation. 

I invite you to find a comfortable seated position, with your back straight, feet planted firmly on the ground and hands gently resting in your lap. Allow your eyes to softly close as you start to unwind.

Bring your attention to your toes; feel the weight of your body, the sensation of the air touching your skin. As you inhale, [PAUSE:2.0] imagine fresh energy entering your toes, filling them with warmth and calmness. Exhale slowly, [PAUSE:4.0] and as you release, allow any tension in your toes to melt away. [PAUSE:2.0] 

Now, bring your awareness to your feet, feeling the connection between your soles and the ground beneath you. As you inhale, [PAUSE:2.0] imagine roots growing from the soles of your feet, deep into the earth, nourishing your entire being. With each exhalation, [PAUSE:4.0] feel your feet becoming heavier, more grounded, releasing any remaining tension.

Gradually, bring your attention to your calves, your knees, your thighs. As you inhale, [PAUSE:2.0] imagine the air filling these areas, soothing any discomfort or tightness. With each exhalation, [PAUSE:4.0] allow your muscles to release, relaxing further with each breath.

Now, bring your focus to your hips, your lower back, and finally your upper back. With each inhale, [PAUSE:2.0] picture the air filling these areas, calming any tension or strain. As you exhale, [PAUSE:4.0] feel your body becoming more relaxed, more tranquil.

As we continue to breathe, allow your entire body to release, to let go of any remaining tension. Imagine yourself standing on a sandy beach at sunset; feel the warmth of the setting sun on your skin, the softness of the sand beneath your feet. [PAUSE:6.0]

Notice how your body feels now, how your muscles are relaxing, releasing their hold on tension. Continue to breathe deeply, feeling your body become heavier, more grounded, more at peace.

As you rest in this calm state, take one final, deep breath in, and when you're ready, slowly open your eyes, feeling refreshed, renewed, and tranquil.""",
"""Hello, welcome to this guided meditation session. I'm here to help you cultivate the energy and motivation you need to tackle your day with confidence.

Find a quiet and comfortable spot to sit or lie down, close your eyes, and take a deep breath in through your nose, and out through your mouth. [PAUSE:2.0]

Imagine a bright, warm light beginning to fill your body, starting at the crown of your head. As this light flows downward, it nourishes your entire being, awakening your senses and revitalizing your spirit. [PAUSE:3.0]

Now, bring your attention to your breath. Feel the sensation of the air moving in and out of your body. As you inhale, imagine fresh, invigorating energy entering your lungs. [PAUSE:1.5] As you exhale, picture any fatigue or stress leaving your body. [PAUSE:2.0]

Envision yourself standing at the edge of a beautiful, tranquil lake. The water's surface is calm and reflective, like a mirror. With each breath, imagine yourself becoming more and more grounded, stable, and centered, just like the stillness of the lake. [PAUSE:3.5]

Now, let's cultivate some inner fire to boost your motivation and productivity. Imagine a spark within your heart, a spark that grows into a warm, gentle flame. As this flame grows, it illuminates your passions and desires, guiding you toward your goals. [PAUSE:4.0]

Visualise yourself tackling your most pressing tasks with ease and confidence. See yourself making progress, overcoming obstacles, and feeling a sense of accomplishment with each step forward. [PAUSE:3.5]

Take a moment to reflect on what you're grateful for in your life. It could be something as simple as a good cup of coffee or a beautiful sunset. Focus on the good things, and allow this sense of gratitude to fill your heart. [PAUSE:3.0]

As you continue to breathe deeply, imagine this energy and motivation spreading throughout your body, infusing every cell with a sense of purpose and drive. [PAUSE:4.0]

When you're ready, slowly open your eyes, and take a moment to notice how you feel. Notice any shifts in your energy, your posture, or your mindset. [PAUSE:2.0]

Remember, this feeling of motivation and productivity is within you, and it's always available. Take it with you throughout your day, and watch how your life transforms.""",
"""We're going to focus on cultivating energy, motivation, and productivity within you. 

Find a comfortable seated position, with your back straight, and feet planted firmly on the ground. 

Close your eyes and take a deep breath in through your nose. [PAUSE:4.0] Exhale slowly through your mouth. [PAUSE:8.0] 

Now, let's introduce a simple yet powerful breathing technique called the '4-7-8 method'. 

Breathe in for four. [PAUSE:4] Hold that life force for seven. [PAUSE:7] Exhale all your tension for eight. [PAUSE:8] 

Repeat this cycle a few more times, allowing yourself to relax and centre. 

As we inhale, imagine fresh, revitalising energy entering your body. [PAUSE:2] Visualise it filling your lungs, your chest, and your entire being. 

As we exhale, imagine any fatigue, stress, or doubts leaving your body. [PAUSE:2] Feel lighter, freer, and more energised with each breath. 

Now, let's visualise a powerful, energising light radiating from the crown of your head. [PAUSE:2] This light represents your inner motivation, drive, and creativity. 

Imagine it flowing down through your body, energising your muscles, your mind, and your heart. [PAUSE:2] Feel it building momentum, filling you with a sense of purpose and determination. 

As you inhale, repeat the phrase 'I am energised, motivated, and productive.' [PAUSE:2] Allow these words to sink deeply into your mind and heart. 

As you exhale, repeat the phrase 'I let go of doubts, fears, and limitations.' [PAUSE:2] Feel a sense of release and freedom with each breath. 

Remember, you have the power to create the life you want. [PAUSE:2] You have the power to choose your thoughts, your emotions, and your actions. 

Now, take one final, deep breath in. [PAUSE:4] Hold it for seven. [PAUSE:7] Exhale slowly, feeling refreshed, renewed, and ready to tackle any challenge that comes your way.""",
"""Hello, welcome to this guided meditation session. I'm here to help you relax and unwind, and guide you through a tranquil walk through nature to help you drift off to sleep.

Find a comfortable seated position, with your back straight and your body relaxed. Close your eyes and take a deep breath in through your nose. [PAUSE:4.0] Hold that breath for a moment, feeling the air fill your lungs. [PAUSE:2.0] Now exhale slowly, allowing any tension to release from your body.

Imagine yourself standing at the edge of a serene forest, surrounded by towering trees that stretch up towards the sky. The air is crisp and clean, filled with the scent of fresh earth and the gentle rustle of leaves.

Take a step forward, and feel the soft earth beneath your feet. [PAUSE:2.0] With each step, allow your body to relax further, letting go of any remaining tension. Feel the weight of your body sinking into the ground, supported by the solid earth beneath you.

As you walk, notice the sounds around you. [PAUSE:2.0] The gentle babbling of a nearby stream, the chirping of crickets, and the soft creaking of branches in the breeze. Allow these soothing sounds to wash over you, calming your mind and body.

Now, imagine a soft, warm light beginning to emanate from the trees above. [PAUSE:2.0] This gentle glow begins to envelop you, filling you with a sense of peace and tranquility. Feel your body relax further, your muscles releasing any remaining tension.

As you continue your walk, notice the sensation of the earth beneath your feet. [PAUSE:2.0] Feel the gentle give of the soil, the softness of the grass, and the solidity of the tree trunks. Allow these sensations to ground you, connecting you to the present moment.

Imagine a soft, fluffy cloud floating gently across the sky. [PAUSE:2.0] Watch as it drifts lazily across the horizon, its softness and lightness a perfect contrast to the earth beneath you. Feel yourself becoming lighter, your body sinking into the ground, supported by the solid earth.

As you continue to walk, notice the sensation of the air on your skin. [PAUSE:2.0] Feel the gentle caress of the breeze, the softness of the leaves, and the warmth of the sun. Allow these sensations to calm your mind and body, washing away any remaining tension.

Now, imagine yourself reaching a tranquil clearing in the forest. [PAUSE:2.0] A place of perfect peace and serenity, surrounded by the soothing sounds of nature. Feel yourself becoming one with the forest, your body relaxing further, your mind calming.

Take one final, deep breath in, feeling the air fill your lungs. [PAUSE:4.0] Hold that breath for a moment, feeling the calm and tranquility spread throughout your body. [PAUSE:2.0] Now exhale slowly, allowing yourself to drift off to sleep, surrounded by the peaceful sounds and sensations of nature.

As you drift off to sleep, remember to breathe deeply, feeling the calm and tranquility spread throughout your body. [PAUSE:2.0] Allow yourself to let go, to release any remaining tension, and to simply be.""",
"""Hello my friend welcome aboard, I'm your skipper for tonight's gentle sail. We're anchored near a majestic lighthouse, standing tall on the rocky shores, the salty sea air fills our sails as the sun sets slowly below the horizon. [PAUSE:3.0]

Imagine the warmth of the fading light on your skin, feel the gentle ocean breeze caressing your face, as you breathe in the salty air, inhale deeply [PAUSE:2.0] and exhale slowly [PAUSE:2.0]. We're drifting now with the current, our sailboat gliding effortlessly through the calm waters.

As we sail closer to the lighthouse, the sound of the waves gently lapping against the shore creates a soothing melody, a lullaby to guide you into a peaceful slumber. [PAUSE:2.0] The light from the lighthouse casts a warm golden glow on the water, as the stars begin to twinkle in the night sky above.

Let's take a moment to focus on our breath, breathe in for four [PAUSE:4.0] hold that life force for seven [PAUSE:7.0] exhale all your tension for eight [PAUSE:8.0]. As you exhale, imagine any stress or worries floating away on the outgoing tide.

Now, imagine yourself standing on the rocky shore, feeling the rough texture of the stone beneath your feet, the cool night air on your skin, the sound of the waves crashing against the shore creating a sense of calm and tranquility. [PAUSE:3.0] As the waves wash over your feet, feel any remaining tension melt away, leaving you feeling peaceful and relaxed.

Let's navigate through these calm waters, our sailboat gliding smoothly, the stars shining brightly above, as we sail towards the setting sun. [PAUSE:2.0] The lighthouse stands watch, a beacon of guidance, a symbol of safety and security.

As we approach the shore, the sound of the waves grows softer, the light from the lighthouse fades into the distance, and we're left with the peaceful silence of the night. [PAUSE:4.0] Take one final deep breath in, hold it for a moment, and exhale slowly, feeling your body relax, your mind calm, and your spirit at peace.

Now, my friend, it's time for you to drift off to sleep, guided by the soothing sounds of the ocean, the gentle rocking of the sailboat, and the peacefulness of the night. [PAUSE:5.0] May your dreams be filled with the wonders of the sea, and may you wake feeling refreshed and renewed.""",
"""Hello and welcome to this guided meditation session. I'm here to support you in cultivating inner peace and calmness, especially in the face of anger.

Find a quiet and comfortable spot to sit, either on a chair, on a cushion on the floor, or even outdoors in nature. Allow your back to relax and elongate, and gently place your feet flat on the ground. Now, take a deep breath in through your nose, and exhale slowly through your mouth.

As you settle, bring your attention to your heart centre. Imagine a bright, pulsing light beginning to emanate from within. With each inhale, envision this light expanding, filling your chest, and radiating outward, calming any tension or stress.

Notice how your body feels in this moment. Allow any areas of physical discomfort to release, letting go with each exhalation. As you breathe, repeat the phrase 'I am calm' to yourself, allowing the words to sink deeply into your being.

[PAUSE:2.0]

Now, bring to mind a situation that typically triggers your anger. It could be a particular person, a specific event, or a certain environment. Visualise this scenario unfolding before you, but this time, imagine yourself responding with calmness and compassion, rather than anger.

[PAUSE:4.0]

As you breathe in, envision fresh air filling your lungs, and with each exhale, imagine any anger or frustration leaving your body. Repeat the phrase 'I choose calm' to yourself, allowing the words to become a gentle reminder.

[PAUSE:3.0]

Imagine a wave of calmness flowing through you, soothing any areas of tension or discomfort. As this wave reaches your heart centre, it grows stronger, filling your entire being with serenity and peace.

[PAUSE:5.0]

Now, bring to mind a loved one, someone you care deeply about. Envision this person standing before you, smiling warmly. Imagine their love and acceptance radiating toward you, filling any gaps or spaces within.

[PAUSE:3.0]

As you inhale, visualise this love and acceptance flowing into your heart, nourishing and soothing any areas of anger or frustration. With each exhale, imagine any residual tension or stress leaving your body.

[PAUSE:4.0]

Remember, anger is a natural emotion, and it's okay to feel it. However, it's how we respond to it that matters. By choosing to cultivate calmness and compassion, you can transform your relationships and interactions with those around you.

Take one final, deep breath in, and as you exhale, repeat the phrase 'I am calm, I am compassionate, and I am at peace.' Allow these words to become a mantra, guiding you forward in your journey toward inner peace and calmness.

[PAUSE:5.0]

As you slowly open your eyes, take a moment to notice how you feel in this present moment. Allow this sense of calmness and compassion to stay with you, guiding you through your daily interactions and relationships.""",
"""Hello and welcome to this extraordinary adventure of self-discovery. As we embark on this thrilling journey, allow yourself to let go of any tension or worries.

Find a comfortable seated position with your back straight, feet planted firmly on the ground. Close your eyes and take a deep breath in through your nose and exhale out through your mouth. [PAUSE:4.0]

Imagine yourself standing in the midst of a bustling city, surrounded by towering skyscrapers and the hum of traffic. As you breathe in, feel the energy of the city pulsing through your veins. Now, with each exhalation, imagine any stress or anxiety leaving your body, like the ebbing of the city's tide.

As you stand there, a whisper in your ear beckons you to leave the city behind and venture into the unknown. You follow the call and find yourself standing at the foot of a majestic mountain range, the snow-capped peaks stretching up towards the sky like giant's fangs.

Take a deep breath, feeling the crisp mountain air fill your lungs. As you inhale, imagine fresh energy and clarity entering your mind. Now, exhale slowly, and with each breath, feel your body relax, your muscles releasing any remaining tension.

You begin your ascent up the mountain, the path winding and treacherous at times. But with each step, you feel a sense of determination and resilience growing within you. You press on, undaunted by the challenges ahead.

As you climb higher, the air grows thinner, and the wind picks up. But you press on, your spirit unbroken. And then, suddenly, you reach the summit, and the breathtaking view takes your breath away.

You see the ocean stretching out before you, a vast expanse of blue that seems to meet the sky at the horizon. The wind whispers secrets in your ear, and you feel a sense of peace and tranquility wash over you.

As you gaze out at the sea, a faint memory stirs within you - a memory of a child, full of wonder and curiosity. This is the treasure you've been seeking all along - a deeper connection to your inner child, a sense of kindness and compassion that has been waiting to be uncovered.

Take a deep breath, and as you exhale, imagine any self-doubt or criticism that may have been holding you back, slowly letting go. You are kind to yourself, just as you would be to a dear friend.

Now, imagine yourself standing on the edge of the desert, the scorching sun beating down upon your skin. But you are not alone - you have the wisdom and resilience of your inner child to guide you.

As you journey across the desert, the sand dunes shifting beneath your feet, you come across a dense jungle, the air thick with the scent of exotic flowers and the sounds of tropical birds. You push through the undergrowth, and as you emerge into a clearing, you see a river flowing gently through the heart of the jungle.

You follow the river, and as you walk, you feel a sense of calm and clarity wash over you. You come across a small wooden boat, and as you step aboard, you feel a sense of trust and surrender. The boat glides smoothly across the water, carrying you back to the city where we first began.

As you return to the city, you feel a sense of completion, a sense of having unearthed a treasure far greater than any material wealth. You feel a sense of kindness and compassion towards yourself, a sense of self-respect that will stay with you for the rest of your journey.

Take one final, deep breath in, and as you exhale, imagine this sense of self-respect and kindness filling you to the brim. You are home, and you are at peace.""",

]


def clear_pauses(txt):
    pattern = r'\[PAUSE:\s?(\d+(?:\.\d+)?)\]'
    return re.sub(pattern=pattern, repl="", string=txt)


split = []

for txt in raw_samples:
    split.extend(recursive_word_chunker(clear_pauses(txt), 15))

messages = []

descriptions = [
    "40-year-old, warm, low pitch, conversational",
    "Female, in her 30s with an American accent and is an event host, energetic, clear diction",
    "Dark villain character, Male voice in their 40s with a British accent. low pitch, gravelly timbre, slow pacing, angry tone at high intensity.",
    "Demon character, Male voice in their 30s with a Middle Eastern accent. screaming tone at high intensity.",
    "Mythical godlike magical character, Female voice in their 30s slow pacing, curious tone at medium intensity.",
]

for s in split:
    prompt = build_prompt(tokenizer, random.choice(descriptions), s)
    messages.append({"text": prompt})


with open("dataset.json", "w") as file:
    file.write(json.dumps(messages[:256], indent=2))

print(json.dumps(messages[:256], indent=2))

print(len(messages))

print(model)