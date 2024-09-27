Example:
Let us assume that we want to improve the quality of the software product and effectiveness of the development process overall. 
We can assume that ”Daily code review increase software quality in comparison to weekly code review” 
To *confirm* or *disprove* this idea we need to check it in practice.
The goal of our experiment, or research objectives, is to *test the effect of different conditions of code inspections on a software quality*.
**Null hypothesis** $H_0$: frequent inspections do not increase the number of detected bugs per kLOC. 
**Alternative hypothesis** $H_1$: frequent inspections increase the number of detected bugs per kLOC.

> We had formed two teams of 9 programmers who conduct inspections for 4 hours at the end of each week. For one of the team we changed the process so they perform daily inspections for 1 hour. Than we were collecting the data about found defects per kLOC for 4 weeks. If we give a treatment to one group only, we can try isolate whether the treatment and *not other factors* influence the outcome.
> At the end we figured out that first team found 140 defects/kLoC, whereas second team found 50 defects/kLoC. Our experiment seems to confirm the alternate hypothesis idea: short frequent code inspections help to increase quality of the software product†. A sound rejection of the null hp requires a statistical test – more later in the course.


---

> **Experiment** is a test under controlled conditions that is made to demonstrate a known truth or examine the validity of a hypothesis.

> **Treatment** (effect, impact) - the exposure of a group to an experimental variable or event, the effects of which are to be measured. 
>            - some impact that we want to check and which to be measured
> **Subjects** - experimental units, who are exposed to treatments. 
> **Observation** - the process of measuring the dependent variable within the experiment. Experimenter performs observation(s) before and/or after the treatment intervention.

> **An experimental design** systematically manipulates one or more variables (frequency and duration of inspections) in order to evaluate how this manipulation impacts an outcome of interest (effectiveness of inspection). An experiment isolates the effects of this manipulation by holding all other variables constant.

### Validity
#### *Internal validity* 
- a measure, that ensures the results and trends seen in an experiment are actually caused by the manipulation (treatment) and not some other factors underlying the process
- the validity of the cause and effect relationship between the independent and dependent variables 
#### *External validity* 
- a measure that shows to which extend the validity of the cause-and-effect relationship is being generalizable to other persons, settings, treatment variables, and measures
- the validity of the cause and effect relationship being generalizable to other settings, treatment variables, and measures 
#### *Construct validity* 
- the validity of inferences about the con- structs (or variables) in the study