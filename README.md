# "System One" classification opportunities for human rights investigations

> [!WARNING]
> This repo is untested. I'm waiting for Jev access so I can test and validate the code. For now, it's my contribution to exploring what this new capability might be useful for.
> If you have access, opinions, or want to contribute, please feel free to create issues in the repo.
> If you have access already and want to contribute by testing and validating, even better. 

Since I started working in human rights, I have kept coming back to the same problem: how do we turn the firehose of information into something people can actually analyse?

Collecting the information is increasingly manageable. We can ingest millions of Telegram posts, social-media messages, articles and other open-source material. Turning that mass of text into something an investigator or researcher can work with is harder.

If I have ten million Telegram posts and want to find gender-based violence, threats, hate speech or potentially inciting language, I can send each message to an LLM with a carefully designed classification prompt. I already do this.

I could also ask it to identify the target, type of threat, protected characteristic and form of abuse, whether violence is advocated, whether the target is an individual or a group, and dozens of other attributes. The question is whether this is a sensible way to classify ten million messages.

## The choices

I could hire people to label the dataset. Fifty researchers classifying 20,000 messages each would get me through the first million. That would take considerable time, along with annotation guidance, quality assurance, adjudication and enough overlapping annotation to check whether the researchers agree.

Human annotation is crucial, especially for legally or contextually complex concepts. But relying on it for every item becomes difficult when the dataset runs to millions or tens of millions of messages. That is where gold-standard labelled subsets can be used for data validation. 

I could use keywords and regular expressions, which are cheap and wonderfully predictable. Embeddings help retrieve similar content; topic modelling and clustering make large datasets easier to explore. I use and like all of these approaches.

A keyword search for *kill*, though, will find threats alongside news reports, condemnations, historical quotations and somebody saying they "killed it" at karaoke. Modern LLMs can resolve much of that ambiguity, at a computational cost.0

> [!TIP]
> It's worth noting here that I'm not making the case for replacing human workers ... ever. This solely refers to the times when there hasn't previously been the manpower or funding to work on such large datasets, and now there is, but we still want to make it as cost effective as possible. 

## Using a language generator as a classifier

A conventional LLM generates sequences of tokens, even when all I need is:

```json
{
  "contains_threat": true
}
```

Depending on the classifier, the model might read a system prompt and taxonomy, inspect the message, reason about it, then produce JSON and an explanation that my software has to parse. That's reasonable for an individual request, but the overhead adds up.

At just 50 output tokens per message, ten million messages require 500 million output tokens. Much of that text may exist only to pass a decision to another piece of software.

The input has overhead too: the same instructions, taxonomy and output schema may be sent repeatedly. Prompt caching and batching help, and smaller models can bring costs down substantially. Even so, I'm paying for language generation when I need a classification.

Autoregressive LLMs generate output one token at a time, conditioned on previous tokens. TypeSafe says Jev can evaluate structured decisions in parallel. Its launch material argues that generating strings is unnecessarily expensive when the output only needs to contain machine-readable decisions.

The computation also requires hardware, electricity and datacentre infrastructure. The environmental footprint depends on the model, provider, hardware and datacentre, so I can't give a useful universal figure for a classification. Across millions of messages, though, reducing the computation per message matters.

I feel uneasy about using AI at this scale. Refusing to use it has some moral appeal, but leaves me with millions of messages that somebody still needs to classify. For now, my somewhat guilt-ridden approach is to use it where I think it has public-interest value, minimise unnecessary computation and measure its limitations.

I'd like to see more progress on efficiency for particular tasks. Aviation is a rough analogy: most passenger flights never became supersonic. Aircraft developed around the speed, efficiency, reliability and cost that made sense for those journeys. I wonder whether some AI workloads need a similar shift.

## What about specialist moderation models?

Meta's Llama Guard family already offers more specialised classification. These models are fine-tuned around safety taxonomies to classify text as safe or unsafe and identify harm categories. Meta describes deriving an "unsafe" probability from the first classification token, which applications can use to set a threshold.

That gets closer to what I need, although a general harmfulness score leaves much of a human-rights research question unanswered.

If a model tells me:

```text
unsafe = 0.93
```

that may help with moderation. For research, I need to know:

> What exactly is present in this message, and which components of a particular human-rights or legal definition does it satisfy?

The score alone doesn't tell me which elements are present. Hateful content may not amount to incitement to violence. A credible threat may not be hate speech. Misogynistic abuse may fall outside the analytical definition being used for technology-facilitated gender-based violence. Advocacy of violence may lack the protected-group and intent elements relevant to genocide.

I want to break these classifications down into observable propositions that can be examined separately.

## Why Jev is interesting

TypeSafe describes Jev as its first "System One Model". It accepts unstructured information and returns predefined, typed probabilistic decisions. The company says those decisions can be evaluated in parallel, with free-form string generation traded for speed, structured outputs and calibrated probabilities.

At launch, TypeSafe listed Jev at $0.042 per million input tokens, with output described as too cheap to meter. It also claimed substantial speed and efficiency gains over conventional LLM workflows for structured decision tasks. These are vendor claims about an early-access system; they need independent testing.

If those claims hold up on real workloads, Jev could make classifying ten million messages affordable for more OSINT and human-rights projects. Its design also gives me a way to ask smaller questions and combine the answers in code.

## Don't ask the model to make the whole decision

Suppose I want to identify direct and public incitement to genocide. I could ask:

```text
Does this message constitute direct and public incitement to genocide?
```

The model might return:

```text
0.82
```

What does `0.82` tell me? The model might be unsure whether the target is a protected group, whether the statement was public, or whether the speaker had the necessary intent. It might have interpreted a euphemism as a direct appeal or missed that the speaker was quoting someone else. I also can't see how it weighted those factors.

I would get more useful information by asking separate questions:

```text
Is the message directed at a national, ethnic, racial or religious group?

Does the speaker advocate killing members of that group?

Does the speaker advocate causing serious bodily or mental harm?

Does the speaker advocate imposing destructive conditions of life?

Would the intended audience understand the statement as a direct appeal?

Was the communication public?

Is there evidence that the speaker intends others to act?

Is there evidence of intent to destroy the group, in whole or in part?
```

Each answer is still an AI judgement and can be wrong. But these smaller, "atomic" propositions give us something we can inspect and combine in ordinary code. Conceptually:

```python
possible_genocide_incitement = (
    protected_group
    and genocidal_act_advocated
    and direct
    and public
    and intent_to_incite
    and genocidal_intent
)
```

The model assesses each proposition; the software defines how the answers combine.

## Binary questions, probabilistic answers

Even a narrowly defined proposition can be ambiguous:

```text
The speaker advocates killing members of the targeted group.
```

The answer need not be a hard `True` or `False`. It could be:

```text
P(True) = 0.97
```

or:

```text
P(True) = 0.54
```

Those probabilities could help the system pass clear positives, reject clear negatives and send ambiguous material to a human analyst. The aim is evidence triage at scale, with legal judgement left to people.

## System One does not mean context-free

Jev's lack of free-form generation or a conventional reasoning stage doesn't restrict its input to the words of a single message. TypeSafe describes that input as structured program state containing unstructured information, so we can supply context alongside the text.

For example:

```json
{
  "message": "...",
  "platform": "Telegram",
  "channel_visibility": "public",
  "subscriber_count": 87000,
  "speaker": "regional political figure",
  "reply_context": "...",
  "previous_messages": ["...", "..."],
  "language_context": "phrase X is commonly used locally as a euphemism for Y"
}
```

With a general-purpose LLM, I can leave a question underspecified and ask the model to reason through it. Here, I need to identify the relevant variables, supply the context, define the questions and decide how the answers interact.

That's more work for whoever designs the classifier. I think it's worth doing: in human-rights research, readers should be able to see and challenge the assumptions in the methodology.

## What this project is trying to build

This repository explores whether Jev and similar models can classify human-rights and OSINT datasets at high volume.

The initial classification families include:

- hate speech;
- advocacy of discrimination, hostility or violence;
- technology-facilitated gender-based violence;
- threats and intimidation;
- dehumanising language;
- direct and public incitement to genocide;
- advocacy of forced displacement;
- attacks or threats against protected persons and objects;
- sexualised abuse and harassment;
- other forms of violent or discriminatory rhetoric.

Questions such as "Is this illegal?" or "Is this a human-rights violation?" involve legal, contextual and evidential judgements that a single classifier score cannot adequately explain.

The project instead asks which observable elements are present, how confident we are about each one, and what follows when we combine them using an explicit analytical framework. The proposed pipeline looks like this:

```text
                       MESSAGE
                          │
                          ▼
                 contextual state
                          │
                          ▼
              ┌─────────────────────┐
              │  atomic AI questions │
              └─────────────────────┘
                 │    │    │    │
                 ▼    ▼    ▼    ▼
               0.98 0.91 0.43 0.07
                 │    │    │    │
                 └────┴────┴────┘
                          │
                          ▼
                 explicit Python logic
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
           classify     review      reject
              │
              ▼
          human analyst
```

My hypothesis is that, for very large text datasets, small probabilistic judgements combined through explicit logic may be more useful than asking a general-purpose LLM to work through the whole classification each time. TypeSafe describes a similar approach: breaking problems into independent questions and using the probabilities in ordinary program logic.

Whether Jev is good enough for human-rights classification still needs testing.

I'm still waiting for access. This repository starts with the methodology, proposed classification schemas and evaluation framework; the architecture alone doesn't tell us whether the model will work.

Once I have access, I'll build gold-standard datasets and compare Jev with human annotations and conventional LLM classifiers. That means measuring false positives and false negatives, testing multilingual and coded language, checking where context changes the classification, and calibrating thresholds. The results should help establish which questions can be automated and which need a human analyst.

I want this work to help researchers examine datasets they otherwise couldn't meaningfully get through, while keeping responsibility for the judgements with the people doing the research.
