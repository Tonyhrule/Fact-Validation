import asyncio
import json
from uuid import uuid4

from helpers.dbscan import cluster
from helpers.oai import async_gpt_calls, get_embeddings

NEWLINE = "\n"

def get_statement_prompt(context: str):
    return f"""Convert the following text into a series of concise, standalone statements.
This should be in bullet-points (- ).
Each statement should be a complete thought and should NOT reference any other bullet points or wider context.
This means that each statement should be able to stand alone and make sense.
For a statement to stand alone, it should contain all the context it needs to be understood.
Each statement MUST include its FULL setting (eg. In xyz, abc happened), and this setting must be specific (eg. Superbowl 50 instead of Superbowl).
ALWAYS use proper nouns if possible. (eg. "Superbowl 50" instead of "the game")

Text:
{context}"""

def get_compress_prompt(cluster: list[str]):
    statements = "\n".join(cluster)
    return f"""Combine the following statements into ONE statement.

Statements:
{statements}"""

async def summarize_two_articles():
    article1 = """Google Reveals Gemini 2, AI Agents, and a Prototype Personal Assistant
    Google once only wanted to organize the world’s information. Now it seems more intent on shoveling that information into artificial intelligence algorithms that become dutiful, ever-present, and increasingly powerful virtual helpers.
    
    Google today announced Gemini 2, a new version of its flagship AI model that has been trained to plan and execute tasks on a user’s computers and the web, and which can chat like a person and make sense of the physical world as a virtual butler.

    “I've dreamed about a universal digital assistant for a long, long time as a stepping stone on the path to artificial general intelligence,” Demis Hassabis, the CEO of Google DeepMind, told WIRED ahead of today’s announcement, alluding to the idea of AI that can eventually do anything a human brain can.

    Gemini 2 is primarily another step up in AI’s intelligence as measured by benchmarks used to gauge such things. The model also has improved “multimodal” abilities, meaning it is more skilled at parsing video and audio and at conversing in speech. The model has also been trained to plan and execute actions on computers.

    “Over the last year, we have been investing in developing more agentic models,” Google’s CEO, Sundar Pichai, said in a statement today. These models, Pichai added, “can understand more about the world around you, think multiple steps ahead, and take action on your behalf, with your supervision.”

    Tech companies believe that so-called AI agents could be the next big leap forward for the technology, with chatbots increasingly taking on chores for users. If successful, AI agents could revolutionize personal computing by routinely booking flights, arranging meetings, and analyzing and organizing documents. But getting the technology to follow open-ended commands reliably remains a challenge, with the risk that errors could translate into costly and hard-to-undo mistakes.

    Still, Google thinks it is moving in the right direction and is introducing two specialized AI agents to demonstrate Gemini 2 agentic potential: one for coding and another for data science. Rather than simply autocompleting sections of code, as current AI tools do, these agents can take on more complex work, such as checking code into repositories or combining data to enable analysis.

    The company is also showing off Project Mariner, an experimental Chrome extension that is capable of taking over web navigation to do useful chores for users. WIRED was given a live demo at Google DeepMind’s headquarters in London. The agent was asked to help plan a meal, which saw it navigate to the website of the supermarket chain Sainsbury’s, log in to a user’s account, and then add relevant items to their shopping basket. When certain items were unavailable, the model chose suitable replacements based on its own knowledge about cooking. Google declined to perform other tasks, suggesting it remains a work in progress.

    “Mariner is our exploration, very much a research prototype at the moment, of how one reimagines the user interface with AI,” Hassabis says.

    Google launched Gemini in December 2023 as part of an effort to catch up with OpenAI, the startup behind the wildly popular chatbot ChatGPT. Despite having invested heavily in AI and contributing key research breakthroughs, Google saw OpenAI lauded as the new leader in AI and its chatbot even touted as perhaps a better way to search the web. With its Gemini models, Google now offers a chatbot as capable as ChatGPT. It has also added generative AI to search and other products.

    Google today also offered a glimpse of how this might transpire with a new version of an experimental project called Astra. This allows Gemini 2 to make sense of its surroundings, as viewed through a smartphone camera or another device, and converse naturally in a humanlike voice about what it sees.

    WIRED tested Gemini 2 at Google DeepMind’s offices and found it to be an impressive new kind of personal assistant. In a room decorated to look like a bar, Gemini 2 quickly assessed several wine bottles in view, providing geographical information, details of taste characteristics, and pricing sourced from the web.

    Through Astra, Gemini 2 can not only search the web for information relevant to a user’s surroundings and use Google Lens and Maps. It can also remember what it has seen and heard—although Google says users would be able to delete data—providing an ability to learn a user’s taste and interests.

    Though the demos were carefully curated, and Gemini 2 will inevitably make errors in real use, the model resisted efforts to trip it up reasonably well. It adapted to interruptions and as WIRED suddenly changed the phone’s view, improvising much as a person might.

    At one point, your correspondent showed Gemini 2 an iPhone and said that it was stolen. Gemini 2 said that it was wrong to steal and the phone should be returned. When pushed, however, it granted that it would be OK to use the device to make an emergency phone call.

    Hassabis acknowledges that bringing AI into the physical world could result in unexpected behaviors. “I think we need to learn about how people are going to use these systems,” he says. “What they find it useful for; but also the privacy and security side, we have to think about that very seriously up front.
    """

    article2 = """Introducing Gemini 2.0: Our New AI Model for the Agentic Era
    Information is at the core of human progress. It’s why we’ve focused for more than 26 years on our mission to organize the world’s information and make it accessible and useful. And it’s why we continue to push the frontiers of AI to organize that information across every input and make it accessible via any output, so that it can be truly useful for you.

    That was our vision when we introduced Gemini 1.0 last December. The first model built to be natively multimodal, Gemini 1.0 and 1.5 drove big advances with multimodality and long context to understand information across text, video, images, audio, and code, and process a lot more of it.

    Today, we are excited to launch Gemini 2.0, our most capable model yet. With new advances in multimodality — like native image and audio output — and native tool use, it will enable us to build new AI agents that bring us closer to our vision of a universal assistant.

    Gemini 2.0 Flash builds on the success of 1.5 Flash, our most popular model yet for developers, with enhanced performance at similarly fast response times. Notably, 2.0 Flash even outperforms 1.5 Pro on key benchmarks, at twice the speed. In addition to supporting multimodal inputs like images, video, and audio, 2.0 Flash now supports multimodal output like natively generated images mixed with text and steerable text-to-speech (TTS) multilingual audio. It can also natively call tools like Google Search, code execution as well as third-party user-defined functions.

    Gemini 2.0 is also available for developers via the Gemini API in Google AI Studio and Vertex AI with multimodal input and text output. A chat-optimized version is also available globally for Gemini users, with further expansion into Google products planned early next year.

    Project Astra, an AI assistant prototype built with Gemini 2.0, is being tested to explore multimodal reasoning and user-friendly interaction. Improvements in Project Astra include better dialogue, enhanced memory capabilities, and new tool use through Google Search, Lens, and Maps. Project Mariner is another research prototype designed for complex web-based tasks, enabling interaction with elements on a browser page to complete user commands.

    In addition, the Gemini 2.0 model powers Jules, an experimental AI-powered code agent, as well as new applications in games, where AI agents assist with planning and strategy in real time.

    We firmly believe in developing AI responsibly, conducting safety assessments, and ensuring transparency as we advance the Gemini family of models and expand agentic possibilities.

    The release of Gemini 2.0 marks a significant step in our journey to create more capable and helpful AI tools. With these advancements, we continue to push the boundaries of what’s possible, bringing us closer to our vision of AI that truly assists in all domains of life.
    """

    raw_contexts = [article1, article2]

    print("Getting statements...")
    new_contexts = [
        str(x).replace("\n- ", "\n").strip().removeprefix("- ")
        for x in await async_gpt_calls(
            [get_statement_prompt(context) for context in raw_contexts],
            progress_bar=True,
        )
    ]

    statements = []
    for statement_list in new_contexts:
        for statement in statement_list.split("\n"):
            if statement.strip() != "":
                statements.append(
                    {
                        "statement": statement.strip(),
                        "id": str(uuid4()),
                    }
                )

    statement_id_to_statement = {s["id"]: s for s in statements}

    print("Getting embeddings...")
    embeddings = await get_embeddings([s["statement"] for s in statements], progress_bar=True)
    for s, e in zip(statements, embeddings):
        s["vector"] = e.vector

    print("Clustering embeddings...")
    clusters = cluster(
        [
            {"vector": e.vector, "id": s["id"]}
            for e, s in zip(embeddings, statements)
        ],
    )

    print("Compressing clusters...")
    no_compress = [c[0] for c in clusters if len(c) == 1]
    to_compress = [c for c in clusters if len(c) > 1]

    compress_prompts = [
        get_compress_prompt([statement_id_to_statement[id]["statement"] for id in cluster])
        for cluster in to_compress
    ]

    compressions = [
        str(x)
        for x in await async_gpt_calls(compress_prompts, progress_bar=True)
    ]

    final_contexts = [
        {
            "id": statement_id_to_statement[statement_id]["id"],
            "content": statement_id_to_statement[statement_id]["statement"],
        }
        for statement_id in no_compress
    ]

    for embedding, cluster_ids, content in zip(compressions, to_compress, compressions):
        final_contexts.append(
            {
                "id": str(uuid4()),
                "content": content,
            }
        )

    all_statements = [fc["content"] for fc in final_contexts]
    total_characters = sum(len(s) for s in all_statements)

    output = {
        "statements": all_statements,
        "character_count": total_characters
    }

    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("Done. Results saved to output.json.")
