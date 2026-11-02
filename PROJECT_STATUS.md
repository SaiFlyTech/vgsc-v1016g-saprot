# Resistance predictor: project status

Predicting pyrethroid-resistance mutations in the *Aedes aegypti* sodium channel

Nov 2, 2026

## Where we are

The system works, and the strongest result is clear: when mutations are judged against Type I pyrethroids, the model ranks every resistant one above every harmless one, perfectly, and a significance test puts the odds of that happening by chance at about one in 1,700. All four databases the project set out to build are complete, the model has been scored against every labelled mutation we could find, and it beats every simpler method by a wide margin. The honest weak spots are known and stated below rather than hidden: Type II pyrethroids are harder, mutation combinations cannot yet be predicted, and the whole exercise rests on thirteen laboratory-measured mutations, which is the real ceiling on what any method here can prove.

## What we set out to do

The question is narrow and practical: given a single change to one amino acid in the mosquito's sodium channel, is that change likely to make the mosquito resistant to pyrethroid insecticides? That channel is the protein pyrethroids act on, and resistance frequently comes from exactly these point changes.

Training a model on examples was never an option, because the examples barely exist. The entire published literature yields thirteen mutations that have been measured in a mosquito channel, and only five of those are confirmed harmless. So instead of training, we use a protein language model, SaProt, that was built from millions of natural protein sequences and knows nothing about insecticides. For each mutation we ask it one thing: how unusual is this change, given everything around it and the shape of the protein there. The assumption is that a change nature almost never permits at that position is more likely to disrupt how the insecticide binds. Because nothing is trained, nothing can be overfitted, which is what makes the result trustworthy on so few labels.

## How the work was built up

The project ran in six phases, each resting on the one before.

1. **Scoping.** Fix the target: the *Aedes aegypti* sodium channel (UniProt C3RTM2), the SaProt model, and a zero-shot approach rather than training.
2. **Groundwork.** Collect the laboratory-measured mutations from the literature and verify every position against the real sequence.
3. **Literature reviews.** Two of them, one on the biology and one on the modelling, to ground the choices and check nothing had been missed.
4. **Pipeline.** Build the chain that turns a mutation name into a score, including the protein's structure.
5. **Databases and scoring.** Build the four databases, score them, and run the evaluation and baselines.
6. **Robustness and reporting.** Stress-test the method, then write up the results and the shortlist.

## The groundwork: curating and verifying the mutations

The quiet but essential part of the project was making sure every mutation was real and correctly placed. The papers number mutations in house-fly convention, which does not match the mosquito sequence, and the offset between the two is not constant. A numbering slip would produce a confident, meaningless score, so every one of the 24 positions was mapped onto the real sequence by alignment and checked against both the wild-type residue and the sequence flanking it. All 24 passed.

Labels were graded by how they were obtained. A mutation counts as resistant or harmless only if it was measured in a cloned channel in the laboratory; being common in resistant mosquitoes does not count, because resistant mosquitoes carry many changes at once. That strict line is what keeps the labelled set small and trustworthy, and it is why the project repeatedly distinguishes a measured result from a field association.

## Two literature reviews

Two reviews were produced and are available as separate documents.

The first covers the biology: every sodium-channel mutation reported in *Aedes aegypti*, what has actually been measured versus merely associated with resistance, the two pyrethroid receptor sites, the combinations, and the haplotypes seen in wild populations. It is the source of the labelled set.

The second covers the modelling: the protein and DNA language models available for this kind of classification, how they are used, which have public code, and how they rank on the standard benchmark. Its main practical finding shaped the road ahead: the model we chose has since been overtaken by newer methods, which is the basis for the recommendation at the end of this document.

## Turning a mutation into a score

The pipeline, for one mutation, is short. Fix and verify the position; cut a window of the protein around it; attach the predicted structure; hide the position and ask the model how unusual the change is. The score is a single number, the difference between the model's confidence in the original amino acid and in the mutant, where a more negative number means more unusual and, by hypothesis, more likely to confer resistance.

Two choices are worth noting. The model reads not just the sequence but the protein's shape, written in a structural alphabet, which is what lets it tell a binding-disrupting change from a harmless one at the same position. And that shape comes from a computational model of the channel, not an experimental structure, which is a real dependency. To make sure the result does not rest on a fragile structure, the project runs a robustness test that deliberately corrupts the structure input and checks the ranking survives. It does, which is what makes the use of a predicted structure defensible.

## The model we use

The project uses a single protein model: **SaProt**. It reads both the amino-acid sequence and the protein's shape, and for each mutation it scores how unusual the change is, with no training. That one model produces every result in this document. SaProt was the best variant-effect model on the standard ProteinGym benchmark when it was chosen, which is why it was picked over a sequence-only model.

Everything else in the pipeline is ordinary supporting machinery, not additional models: a small tool that prepares the structure input, a predicted shape of the channel, and two simple yardsticks, a standard substitution table and a conservation count, used only to show that SaProt beats them. Nothing is trained. Scoring needs a GPU for a few minutes, which a free cloud notebook provides, and every other step runs on an ordinary laptop.

## The four databases, and how each turned out

| # | Database | What it holds | Outcome |
| --- | --- | --- | --- |
| 1 | Resistance set and controls | 25 single mutations and 8 combinations with laboratory results | Built and scored. Delivered the headline result below |
| 2 | Every amino acid at each position | All 456 single changes across the 24 positions | Built and scored. Delivered a 42-mutation shortlist for the laboratory |
| 3 | Combinations | Field haplotypes and measured combinations | Built and scored. Gave one new prediction; interactions remain unpredictable |
| 4 | Aedes sequences from UniProt | Reference protein sequences, not mutations | Built and examined. Exhausted: holds no new usable data |

**Database 1** is the only one with laboratory answers, so it is the only one the model can be judged against. Its trainable core is thirteen single mutations measured in mosquito channels, eight resistant and five harmless, plus eight combinations.

**Database 2** cost almost nothing to score, because one pass of the model returns every amino acid at a position at once, so all 456 changes came from 24 passes. Its value is a confidence scale and a shortlist of untested changes worth measuring.

**Database 3** scored the real mosquito haplotypes. It confirmed the single-mutation picture and produced one genuinely new prediction, but the model cannot yet judge how two mutations interact, for a structural reason explained below.

**Database 4** is different in kind: a set of whole protein sequences, not mutations, meant to supply natural variation. On examination the *Aedes aegypti* set is the reference gene and its isoforms, with no population-level diversity, so it yields no new harmless controls. That is a genuine finding, not a gap in the work: the diversity we would want lives in genome databases, not UniProt.

## The results

Splitting by insecticide class, which the client asked for, is what shows the model at its best. The reason is biological: most of these mutations affect only one of the two pyrethroid types, so a single resistant-or-not verdict throws away the part a control programme would act on.

| What is asked | Ranking | Honest accuracy | Significance |
| --- | --- | --- | --- |
| Type I pyrethroids | 1.00 | 0.92 (12 of 13) | p = 0.0006 |
| Both classes at once | 0.91 | 0.85 (11 of 13) | p = 0.0062 |
| Type II pyrethroids | 0.79 | 0.62 (8 of 13) | p = 0.048 |
| BLOSUM62, a standard baseline | about 0.65 | about 0.69 | — |

Ranking asks whether resistant mutations score below harmless ones, where 1.00 is perfect and 0.50 is chance. Accuracy is the harder yes/no verdict, measured properly by fixing the cut-off without ever seeing the mutation it is tested on, and splitting the data by position rather than by mutation so the model cannot score well for the wrong reason.

Two honest notes belong with these numbers. Type II is the genuine weak spot, because a score that only measures how unusual a change is cannot know which of the two insecticides it spares. And an earlier informal figure of 11 out of 11 has been retired: it was computed by choosing the cut-off after seeing the scores, and the proper figure is 11 of 13 overall, 12 of 13 for Type I. Those are the numbers to quote.

## What it was measured against

A good score means nothing without something to beat, so the model was tested against both a dumb baseline and a serious one.

| Method | Mosquito-tested | All species |
| --- | --- | --- |
| SaProt, our model | 1.00 | 0.75 |
| BLOSUM62 (dumb lookup) | 0.68 | 0.60 |
| Smarter conservation (PSSM) | 0.61 | 0.45 |
| Simple conservation count | 0.62 | 0.39 |

The serious competitor is conservation: for each mutation, look at the same position across many other insect sodium channels and ask how often that change appears in nature. We ran it in a simple and a smarter form, the smarter one being the standard textbook method. Both lost, and not narrowly; the smarter version does not even beat the dumb BLOSUM lookup.

This matters more than a win on paper. It shows the model is not merely rediscovering conservation. Conservation can only see whether a position is conserved, so it wrongly flags harmless changes at conserved spots; the two harmless controls D1763Y and T1520I score as more unusual by conservation than most resistant mutations do. The model, with the protein's structure and the specific amino acid in view, tells them apart. That is the gap a sceptical reviewer would probe, and the answer is now on record.

## We tested whether training helps

The obvious question is whether a trained model would beat the untrained score. We tested it two ways, both trained only on database 1's labels with the same honest protocol: first a classifier on the model's scores, then a classifier on the model's own internal representations, its embeddings. Neither helps.

Training on the scores:

| Task | Untrained score | Trained on the score | Trained on three features |
| --- | --- | --- | --- |
| Type I | 1.00 | 1.00 | 0.98 |
| Both classes | 0.91 | 0.86 | 0.79 |
| Type II | 0.79 | 0.69 | 0.59 |

The second test is the stronger one, and the one usually meant by "training the protein model": it learns from the 1,280-number vector the model produces for each mutation. With only thirteen examples this overfits catastrophically, and because the evaluation leaks nothing, the damage is visible rather than hidden.

| Task | Untrained score | Trained on the model's embeddings |
| --- | --- | --- |
| Type I | 1.00 | 0.64 |
| Both classes | 0.91 | 0.56 |
| Type II | 0.79 | 0.04 |

Several of the trained scores fall below 0.50, worse than chance, which is the signature of a model fitting noise instead of signal. A leak-free evaluation is what exposes this; a careless one would have looked deceptively good. Fine-tuning the model's weights directly would be the same problem, larger, so it is not attempted. The conclusion is now firm and doubly tested: on this much data the untrained score is the model, and the way forward is more measured mutations or a better untrained model, not training. The trained pipeline itself has been built and checked, and is ready for the day the labelled set is large enough to use it.

## What the project now produces

Beyond the evaluation, two outputs are usable in their own right.

**A shortlist for the laboratory.** From the full sweep of 456 possible changes, 42 stand out: they score in the same band as every known resistant mutation, and they sit in a pyrethroid binding site, so disruption there is mechanically plausible. Most useful are the ones at positions already proven to matter (1007, 1011, 1016, 1534), where only a different amino acid is being proposed. The most extreme changes were deliberately excluded, because a change that drastic likely just breaks the channel rather than conferring resistance. This is the most actionable thing a model can produce while laboratory labels are scarce: a ranked set of experiments worth running.

**A prediction for a circulating haplotype.** Of the mutation combinations found in wild mosquitoes, all but one already have laboratory data. The exception, S989P with V1016G and D1763Y, found in Taiwan, scores essentially identical to S989P with V1016G alone. The model's claim is therefore specific and testable: adding D1763Y to that haplotype changes nothing, which fits the biology, since D1763Y is a known no-effect mutation.

## The honest limits

Stated plainly, because they bound what the system can claim.

- **The channel is one route to resistance, not the only one.** A mosquito can be strongly resistant through other mechanisms with a perfectly ordinary channel, so a clean score is never a verdict that the mosquito is susceptible.
- **Thirteen labels.** All five harmless controls come from a single research group, and no model can manufacture more data. This is the true ceiling.
- **Combinations cannot yet be predicted.** The model reads a limited window of the protein, and most real haplotypes carry mutations too far apart to sit in one window, so it cannot see them interact.
- **Type II is weaker than Type I,** for the structural reason given above.
- **The structure is predicted, not measured,** so the project keeps a noise test that corrupts the structure input and checks the ranking survives, which it does.

## What comes next

One clear technical move stands out. The model we use was the best variant-effect model on the standard benchmark when we chose it, but it has since been overtaken by methods that combine a homology alignment with structure rather than structure alone. The current leader, VenusREM, is a strong fit: it needs no training, it runs on the free hardware we already use, and it needs exactly the four inputs we already hold. It is also the most promising route past the window limit that currently blocks combination prediction. Trying it is the natural next phase.

Two decisions sit with the client. First, whether more harmless laboratory labels can be found, from unpublished work or collaborators, since that, more than any model change, is what would lift the ceiling. Second, whether the search for natural harmless variants should move from UniProt, which is now exhausted, to genome and population databases, which is a change of scope rather than another download.

For completeness, two running tasks remain open and low-stakes: a biologist's review of the mutation labels and the house-fly-to-mosquito position mapping, and confirmation of the required accuracy for the platform to be useful in the field. Neither blocks the work; both would sharpen it.
