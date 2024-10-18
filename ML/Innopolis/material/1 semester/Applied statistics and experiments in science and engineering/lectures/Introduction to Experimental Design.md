Example:
Let us assume that we want to improve the quality of the software product and effectiveness of the development process overall. 
We can assume that ”Daily code review increase software quality in comparison to weekly code review” 
To *confirm* or *disprove* this idea we need to check it in practice.
The goal of our experiment, or research objectives, is to *test the effect of different conditions of code inspections on a software quality*.
**Null hypothesis** $H_0$: frequent inspections do not increase the number of detected bugs per kLOC. 
**Alternative hypothesis** $H_1$: frequent inspections increase the number of detected bugs per kLOC.

> We had formed two teams of 9 programmers who conduct inspections for 4 hours at the end of each week. For one of the team we changed the process so they perform daily inspections for 1 hour. Than we were collecting the data about found defects per kLOC for 4 weeks. If we give a treatment to one group only, we can try isolate whether the treatment and *not other factors* influence the outcome.
> At the end we figured out that first team found 140 defects/kLoC, whereas second team found 50 defects/kLoC. 
> Our experiment seems to confirm the alternate hypothesis idea: short frequent code inspections help to increase quality of the software product†. A sound rejection of the null hypothesis requires a statistical test.
---

> **Experiment** is a test under controlled conditions that is made to demonstrate a known truth or examine the validity of a hypothesis.
> **Treatment** (effect, impact) - the exposure of a group to an experimental variable or event, the effects of which are to be measured. 
>            - some impact that we want to check and which to be measured
> **Subjects** - experimental units, who are exposed to treatments. 
> **Observation** - the process of measuring the dependent variable within the experiment. Experimenter performs observation(s) before and/or after the treatment intervention.
> **An experimental design** systematically manipulates one or more variables (frequency and duration of inspections) in order to evaluate how this manipulation impacts an outcome of interest (effectiveness of inspection). An experiment isolates the effects of this manipulation by holding all other variables constant.

> **Sample** – finite set of objects from the population.
> ***Types of sample:***
> **Random sample** - each item in the population is selected randomly, i.e., informally has an equal probability of being selected.
> **Convenience sample** - items are chosen on their convenience and availability (only few persons agreed to participate in our experiment)

### Characteristics of experiment
*(fully fledged experiment has the following key characteristics):*
- **Random assignment** - is the process of assigning individuals at random to groups or to different groups in an experiment. (Random assignment != Random selection) (**Random selection** - is the process of selecting a sample from a population, so that the sample is representative of the population and you can generalize results obtained during the study to the population). *(Experiments may not include random selection; however, the most sophisticated type of experiment should involve random assignment.)*
- **Control over extraneous factors** by implementing: pretest&posttest, covariates, matching of participants, homogeneous samples, blocking variables.
	- **Extraneous factors** - are any influences in the selection of the participants, the procedures, the statistics, or the design likely to *affect the outcome* and provide an alternative explanation for our results than what we expected. 
	- **Pretest** - provide a measure on some attribute or characteristic that you assess for participants in an experiment before they receive a treatment. Pretest *may affect aspects of the experiment*, they are often statistically controlled for by using the procedure of **covariance** rather than by simply comparing them with posttest scores.
	- **Posttest** - is a measure on some attribute or characteristic that is assessed for participants in an experiment after a treatment.
	- **Covariates** - are variables that the researcher controls for using statistics and that *relate to the dependent* variable but that don't relate to the independent variable. *The statistical procedure of covariance removes the variance shared by the covariate and the dependent variable, so that the variance between the independent and dependent variable (plus error) is all that remains.*
	- **Matching** - is the process of identifying one or more characteristics that *influence the outcome* and assigning items with that characteristics *equally* to the experimental and control groups.![[Matching.png|300]]
	- **Homogeneous sampling** - is selecting for experimental and control groups items which vary little in their characteristics. *The more similar they are in personal characteristics or attributes, the more these characteristics or attributes are **controlled in the experiment**.*
	- **Blocking variable** - is a variable the researcher controls before the experiment starts by dividing (or “blocking”) the items into subgroups (or categories) and analyzing the impact of each subgroup on the outcome. *In this procedure, the researcher forms homogeneous by choosing a characteristic common to all participants in the study. Then the researcher randomly assigns individuals to the control and experimental groups using each category of the variable.*
- **Manipulation of the treatment conditions** - treatment variables need to have two or more categories, or levels. The experimental researcher manipulates one or more of the treatment variable conditions.
	- In experiments, you need to focus on the independent variables. These variables influence or affect the dependent variables in a quantitative study.
	- **Treatment variables** are independent variables that the researcher manipulates to determine their effect on the dependent variable. 
	- *In our example the treatments are different levels of frequency or duration of inspections*
- **Outcome measures**
- **Group comparisons**
- **Threats to validity** - experiments might lead to wrong conclusion, meaning that the results would not be “valid”
  https://www.scribbr.com/methodology/internal-validity/
	- **Internal validity** - a measure, that ensures the results and trends seen in an experiment are actually caused by the manipulation (treatment) and not some other factors underlying the process (*communication across the teams*).
		- **History** - any event that occurs while the experiment is in progress. *Might be an alternation; using a control group mitigates this concern.* Refers to the possibility that specific events, other than the intended treatment, may have occurred between the pretest and post-test observations and may obscure the true treatment effect. 
		- **Regression** - the natural tendency for extreme scores to regress or move towards the mean. Operating where groups have been selected on the basis of their *extreme scores*. 
		- **Mortality** - if groups lost participants (e.g., due to dropping out of the experiment) they may not be equivalent.
		- **Diffusion of treatments** - when the experimental and control groups can communicate with each other, the control group may learn from the experimental group information about the treatment.
		- **Compensatory rivalry** - if you publicly announce assignments to the control and experimental groups, compensatory rivalry may develop between the groups because the control group feels that it is the “underdog”
		- **Testing** - a pretest may confound the influence of the experimental treatment; using a control group mitigates this concern
		- **Maturation** - normal changes over time (by natural) (not specific to the particular events) (e.g., fatigue, aging or improving skill) might affect the dependent variable; using a control group mitigates this concern. 
		- **Selection** - if randomization is not used to assign participants, the groups may not be equivalent
		- **Interactions with selection** - a bias in selection may produce subjects that are more or less sensitive to the experimental treatment. Several of the threats mentioned above can interact (or relate) with the selection of participants to add additional threats to an experiment. 
		- **Compensatory equalization** - when only the experimental group receives a treatment, an inequality exists that may threaten the validity of the study. The benefits of the experimental treatment need to be equally distributed among the groups in the study
		- **Resentful demoralization** - when a control group is used, individuals in this group may become resentful and demoralized because they perceive that they receive a less desirable treatment than other groups.
		- **Instrumentation** - changes in the calibration of a measuring instrument or changes in the observers or scorers used may produce changes in the obtained measurements
	- **External validity** - a measure that shows the validity of the extent to which the results of a study can generalize to other situations, people, settings, treatment variables, and measures.
	  https://www.scribbr.com/methodology/external-validity/
		- **Reactive effect of testing** - pretest might increase or decrease the respondent’s sensitivity or responsiveness to the experimental variable. *The conducting of a pre- or post-test affects the outcomes.*
		- **Interaction of selection and treatment** - the inability to generalize beyond the groups in the experiment. *Factors like the setting, time of day, location, researchers’ characteristics, etc. limit generalizability of the findings.*
		- **Reactive effects of experimental arrangements** - inability to generalize from the setting where the experiment occurred to another setting.
		- **Multiple-treatment interference** - occurs when treatments are applied to the same respondents, but the effects of prior treatments are not usually erasable
		- **Interaction of history and treatment** - when the researcher tries to generalize findings to past and future situations
	- **Construct validity** - is about how well a test measures the concept it was designed to evaluate. (refers informally to whether I apply the “right” analysis, that if, for instance, if I apply the mean on data on an ordinal scale. Formally, *it defines whether the operational definition of a variable actually reflect the true* *theoretical* meaning of a concept. In other words, whether a scale or test measures the construct adequately.)
	  https://www.scribbr.com/methodology/construct-validity/
		- **Convergent validity** tests that confirms that are expected to be related are, in fact, related. 
		- **Divergent validity** tests that confirms that should have no relationship do, in fact, not have any relationship.