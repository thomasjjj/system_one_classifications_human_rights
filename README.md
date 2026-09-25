# System One classification opportunities for human rights

> [!WARNING]
> It should be noted that this repo is untested while I wait for Jev access to test and validate the code. Treat it as my contribution to the discovery phase of the new capability where everyone is scrambling to identify all of its use cases. 

Since I started working in the human rights sector, I have been repeatedly trying to solve some version of the same problem: **how do we turn the firehose of information into something that humans can actually analyse?**

Collecting information is increasingly not the difficult part. It is perfectly possible to ingest millions of Telegram posts, social-media messages, articles and other pieces of open-source information. The harder problem is transforming that mass of text into structured information that an investigator, researcher or analyst can work with.

If I have ten million Telegram posts in a database and want to know which contain gender-based violence, threats, hate speech or potentially inciting language, I can technically do that today. I can send every message to an LLM, give it a carefully designed prompt and ask it to return a classification. In fact, I do that now.

I can go further. I could ask it to identify the target, type of threat, protected characteristic, form of abuse, whether violence is advocated, whether a statement is directed at an individual or group, and dozens of other attributes.

The problem is not whether this is possible.

The problem is whether it is a sensible way to classify ten million messages.

## The unpleasant choices

There are several ways I could approach a dataset of that size.

I could hire people to label it manually. Fifty researchers could each painstakingly classify 20,000 messages and, after a considerable amount of time, I would have labelled my first million. I would also need annotation guidance, quality assurance, adjudication and enough overlapping annotation to measure whether the researchers actually agree with each other.

Human annotation remains extremely important, particularly when dealing with legally or contextually complex concepts, but it is difficult to make it the primary classification mechanism for datasets containing millions or tens of millions of items.

I could use traditional computational methods. Keywords and regular expressions are cheap and wonderfully predictable. Embeddings can retrieve semantically similar content. Topic modelling and clustering can reduce large datasets into more manageable groups. In practice, I use and like all of these approaches.

But they have limitations. A keyword can find the word *kill*; it is much less capable of distinguishing between someone threatening to kill a group, a journalist reporting that threat, a person condemning it, a historical quotation and somebody saying that they "killed it" at karaoke.

Modern LLMs solve a great deal of this ambiguity.

Unfortunately, they solve it rather extravagantly.

## Using a language generator as a classifier

A conventional LLM is fundamentally a system for generating sequences of tokens. Even when I only need something resembling:

```json
{
  "contains_threat": true
}
```

I am still invoking a model designed to generate language. 

Depending on how I build the classifier, I might ask it to read a system prompt, read a classification taxonomy, inspect the message, reason about the answer, produce JSON, explain its reasoning and perhaps wrap the whole thing in a schema that my software then parses.

For an individual request, this is perfectly reasonable.

At large scale, small inefficiencies compound.

If a classifier generates only **50 output tokens per message**, running it over ten million messages means generating **500 million output tokens**. If most of those tokens exist purely to communicate a decision to another piece of software, much of that generation is effectively scaffolding around the thing I actually wanted: a decision.

There is also repeated input overhead. The model may need to receive versions of the same instructions, taxonomy and output schema again and again. Prompt caching and batching can mitigate this, and smaller models make the economics substantially better, but the underlying architecture is still being asked to perform a task that looks somewhat different from what it was principally designed to do.

Autoregressive LLMs produce their outputs sequentially: one token is generated conditioned on the previous tokens. TypeSafe contrasts this with Jev's model, where a set of structured decisions can be evaluated in parallel rather than emitted as a piece of generated text. TypeSafe's launch material explicitly argues that strings are extremely flexible but unnecessarily expensive when the desired output is a collection of machine-readable decisions.

There is a resource question here too. Inference requires hardware, electricity and supporting datacentre infrastructure. The exact environmental footprint of a classification depends on the model, hardware, provider and datacentre and is difficult to reduce to a meaningful universal figure. But the general principle is obvious: if I want to evaluate millions of messages, making each evaluation cheaper computationally matters.

This leaves me in a slightly uncomfortable position. I could take a principled stand against using AI for this kind of mass classification, which has a certain moral appeal, but I would still have millions of messages that somebody needs to classify.

So, for now, I have taken the more guilt-ridden approach: use the technology where it produces meaningful public-interest value, try to minimise unnecessary computation, measure its limitations, and hope that progress increasingly shifts from simply making models *bigger* towards making particular forms of machine intelligence dramatically more efficient.

Aviation provides a rough analogy. Technological progress did not result in every passenger flight becoming supersonic. For most journeys we converged on aircraft optimised around a much more useful combination of speed, efficiency, reliability and cost.

For some AI workloads, perhaps the equivalent transition is overdue.

## What about specialist moderation models?

There are already models designed more specifically for classification.

Meta's **Llama Guard** family is an obvious example. Llama Guard models are fine-tuned around safety taxonomies and can classify text as safe or unsafe and identify relevant harm categories. Meta describes obtaining an "unsafe" probability from the probability assigned to the model's first classification token, allowing applications to threshold that score.

This is considerably closer to what I want than asking a general-purpose chatbot to write an essay about every Telegram post.

But it exposes another problem.

A **generic harmfulness score is not the same thing as a human-rights classification**.

If a model tells me:

```text
unsafe = 0.93
```

that might be extremely useful for content moderation. It is much less useful if my research question is:

> What exactly is present in this message, and which components of a particular human-rights or legal definition does it satisfy?

There are too many potentially hidden variables inside the score.

A piece of content can be extremely hateful without amounting to incitement to violence. It can contain a credible threat without being hate speech. It can contain misogynistic abuse without meeting the analytical definition being used for technology-facilitated gender-based violence. A statement can advocate violence against people while lacking the specific protected-group and intent elements relevant to genocide.


The purpose of this project is therefore not to create another universal **badness score**.

It is to decompose complicated classifications into observable propositions.

## Why Jev is interesting

This is where TypeSafe's **Jev** becomes particularly interesting.

TypeSafe describes Jev as its first "System One Model": rather than generating arbitrary strings, Jev accepts unstructured information and returns predefined, typed probabilistic decisions. Its architecture is intended specifically for automation and high-volume decision workloads. TypeSafe says the individual decisions can be evaluated in parallel and describes the model as giving up free-form string generation in exchange for speed, structured outputs and calibrated probabilities.

At launch, TypeSafe listed Jev at **$0.042 per million input tokens**, with output described as too cheap to meter, while claiming substantially greater speed and efficiency than conventional LLM workflows on the kinds of structured decision tasks it is designed for. These are currently vendor claims from an early-access system and need independent testing, but if the economics survive contact with real workloads they are extremely interesting for OSINT and human-rights research.

Ten million classifications stops looking quite so absurd when the primitive being purchased is closer to a very cheap probabilistic decision than a miniature conversation with a language model.

More importantly, the design of Jev encourages a different way of thinking about the problem.

## Don't ask the model to make the whole decision

Imagine that I want to identify **direct and public incitement to genocide**.

The tempting approach is:

```text
Does this message constitute direct and public incitement to genocide?
```

And perhaps the model returns:

```text
0.82
```

But what does `0.82` actually mean?

Was it unsure whether the target constituted a protected group?

Was it confident the language advocated killing but uncertain whether it was public?

Did it detect genocidal rhetoric but not the necessary intent?

Did it interpret a euphemism as a direct call to violence?

Was the speaker quoting somebody else?

Did the model silently weight one factor more heavily than another?

A single score conceals all of this.

Instead, the much more interesting approach is to ask a collection of much smaller questions:

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

These are still AI judgements, and they can still be wrong.

But each question is much closer to an **atomic proposition**.

Instead of asking the model to somehow encode an entire legal test inside a mysterious `0.82`, we ask it a collection of propositions whose meaning we understand and combine them ourselves in ordinary code.

Conceptually:

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

The AI helps answer fuzzy questions.

**The software controls the logic.**



## Binary questions, probabilistic answers

"Atomic" does not mean pretending ambiguity has disappeared.

The questions can be framed as binary propositions:

```text
The speaker advocates killing members of the targeted group.
```

but the useful output is not necessarily a hard `True` or `False`.

It may instead be:

```text
P(True) = 0.97
```

or:

```text
P(True) = 0.54
```

That uncertainty is valuable.

A system can confidently pass straightforward examples, reject straightforward negatives and route ambiguous material towards a human analyst.

The resulting architecture begins to look less like automated legal judgement and more like a **large-scale evidence triage system**.

That is a much more useful goal.

## System One does not mean context-free

There is an important limitation to understand.

Jev's lack of free-form generation or a conventional reasoning stage does not mean it is restricted to the literal words contained in a message. TypeSafe describes the input as structured program state containing unstructured information. We can therefore supply contextual information alongside the text.

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

What changes is **where the reasoning architecture lives**.

A general-purpose LLM can be handed an underspecified question and asked to reason its way toward an answer.

With the approach explored here, we should identify the variables ourselves, provide the relevant state, ask narrowly defined questions and explicitly define how their answers interact.

That requires more work from the person designing the classifier.

I consider that a feature.

For human-rights research, important assumptions should ideally be visible in the methodology rather than buried somewhere inside a prompt or an invisible chain of reasoning.

## What this project is trying to build

This repository explores whether Jev and similar decision-oriented models can be used as a high-volume classification layer for human-rights and OSINT datasets.

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

The aim is not to ask an AI system:

> **Is this illegal?**

Nor is it:

> **Is this a human-rights violation?**

Those questions frequently contain legal, contextual and evidential judgements that should not be collapsed into a single opaque classifier score.

Instead, the project asks:

> **What observable elements are present in this material, how confident are we that each element is present, and what conclusions follow when we combine those elements using an explicit analytical framework?**

That produces a pipeline resembling:

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

The hypothesis is straightforward:

**for very large text datasets, a collection of small, explicit, probabilistic judgements may be more useful than repeatedly asking a general-purpose language model to reason through and generate the answer to a large classification problem.**

TypeSafe makes a similar argument in its own description of Jev: its more reliable workflows tend to decompose problems into many independent questions and then use the resulting probabilities inside ordinary program logic.

Whether Jev specifically is good enough for human-rights classification is an empirical question.

At the time of writing I am still waiting for access, so this repository begins with the methodology, proposed classification schemas and evaluation framework rather than assuming the model works simply because its architecture is appealing.

Once access is available, the interesting work begins: build gold-standard datasets, compare Jev against human annotations and conventional LLM classifiers, measure false positives and false negatives, test multilingual and coded language, determine where context changes classifications, calibrate thresholds, and work out which questions can genuinely be automated and which should remain firmly in the hands of human analysts.

That, ultimately, is the point of the project.

Not to replace human judgement with a model.

To make it possible for human judgement to operate over datasets that would otherwise be impossible to meaningfully examine.