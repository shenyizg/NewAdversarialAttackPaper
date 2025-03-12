# Latest Adversarial Attack Papers
**update at 2025-03-12 19:26:14**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. Birds look like cars: Adversarial analysis of intrinsically interpretable deep learning**

cs.LG

Preprint

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2503.08636v1) [paper-pdf](http://arxiv.org/pdf/2503.08636v1)

**Authors**: Hubert Baniecki, Przemyslaw Biecek

**Abstract**: A common belief is that intrinsically interpretable deep learning models ensure a correct, intuitive understanding of their behavior and offer greater robustness against accidental errors or intentional manipulation. However, these beliefs have not been comprehensively verified, and growing evidence casts doubt on them. In this paper, we highlight the risks related to overreliance and susceptibility to adversarial manipulation of these so-called "intrinsically (aka inherently) interpretable" models by design. We introduce two strategies for adversarial analysis with prototype manipulation and backdoor attacks against prototype-based networks, and discuss how concept bottleneck models defend against these attacks. Fooling the model's reasoning by exploiting its use of latent prototypes manifests the inherent uninterpretability of deep neural networks, leading to a false sense of security reinforced by a visual confirmation bias. The reported limitations of prototype-based networks put their trustworthiness and applicability into question, motivating further work on the robustness and alignment of (deep) interpretable models.



## **2. Beyond Optimal Fault Tolerance**

cs.DC

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2501.06044v4) [paper-pdf](http://arxiv.org/pdf/2501.06044v4)

**Authors**: Andrew Lewis-Pye, Tim Roughgarden

**Abstract**: The optimal fault-tolerance achievable by any protocol has been characterized in a wide range of settings. For example, for state machine replication (SMR) protocols operating in the partially synchronous setting, it is possible to simultaneously guarantee consistency against $\alpha$-bounded adversaries (i.e., adversaries that control less than an $\alpha$ fraction of the participants) and liveness against $\beta$-bounded adversaries if and only if $\alpha + 2\beta \leq 1$.   This paper characterizes to what extent "better-than-optimal" fault-tolerance guarantees are possible for SMR protocols when the standard consistency requirement is relaxed to allow a bounded number $r$ of consistency violations. We prove that bounding rollback is impossible without additional timing assumptions and investigate protocols that tolerate and recover from consistency violations whenever message delays around the time of an attack are bounded by a parameter $\Delta^*$ (which may be arbitrarily larger than the parameter $\Delta$ that bounds post-GST message delays in the partially synchronous model). Here, a protocol's fault-tolerance can be a non-constant function of $r$, and we prove, for each $r$, matching upper and lower bounds on the optimal "recoverable fault-tolerance" achievable by any SMR protocol. For example, for protocols that guarantee liveness against 1/3-bounded adversaries in the partially synchronous setting, a 5/9-bounded adversary can always cause one consistency violation but not two, and a 2/3-bounded adversary can always cause two consistency violations but not three. Our positive results are achieved through a generic "recovery procedure" that can be grafted on to any accountable SMR protocol and restores consistency following a violation while rolling back only transactions that were finalized in the previous $2\Delta^*$ timesteps.



## **3. Low-Cost Privacy-Preserving Decentralized Learning**

cs.LG

24 pages, accepted at Pets 2025

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2403.11795v3) [paper-pdf](http://arxiv.org/pdf/2403.11795v3)

**Authors**: Sayan Biswas, Davide Frey, Romaric Gaudel, Anne-Marie Kermarrec, Dimitri Lerévérend, Rafael Pires, Rishi Sharma, François Taïani

**Abstract**: Decentralized learning (DL) is an emerging paradigm of collaborative machine learning that enables nodes in a network to train models collectively without sharing their raw data or relying on a central server. This paper introduces Zip-DL, a privacy-aware DL algorithm that leverages correlated noise to achieve robust privacy against local adversaries while ensuring efficient convergence at low communication costs. By progressively neutralizing the noise added during distributed averaging, Zip-DL combines strong privacy guarantees with high model accuracy. Its design requires only one communication round per gradient descent iteration, significantly reducing communication overhead compared to competitors. We establish theoretical bounds on both convergence speed and privacy guarantees. Moreover, extensive experiments demonstrating Zip-DL's practical applicability make it outperform state-of-the-art methods in the accuracy vs. vulnerability trade-off. Specifically, Zip-DL (i) reduces membership-inference attack success rates by up to 35% compared to baseline DL, (ii) decreases attack efficacy by up to 13% compared to competitors offering similar utility, and (iii) achieves up to 59% higher accuracy to completely nullify a basic attack scenario, compared to a state-of-the-art privacy-preserving approach under the same threat model. These results position Zip-DL as a practical and efficient solution for privacy-preserving decentralized learning in real-world applications.



## **4. Adv-CPG: A Customized Portrait Generation Framework with Facial Adversarial Attacks**

cs.CV

Accepted by CVPR-25

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2503.08269v1) [paper-pdf](http://arxiv.org/pdf/2503.08269v1)

**Authors**: Junying Wang, Hongyuan Zhang, Yuan Yuan

**Abstract**: Recent Customized Portrait Generation (CPG) methods, taking a facial image and a textual prompt as inputs, have attracted substantial attention. Although these methods generate high-fidelity portraits, they fail to prevent the generated portraits from being tracked and misused by malicious face recognition systems. To address this, this paper proposes a Customized Portrait Generation framework with facial Adversarial attacks (Adv-CPG). Specifically, to achieve facial privacy protection, we devise a lightweight local ID encryptor and an encryption enhancer. They implement progressive double-layer encryption protection by directly injecting the target identity and adding additional identity guidance, respectively. Furthermore, to accomplish fine-grained and personalized portrait generation, we develop a multi-modal image customizer capable of generating controlled fine-grained facial features. To the best of our knowledge, Adv-CPG is the first study that introduces facial adversarial attacks into CPG. Extensive experiments demonstrate the superiority of Adv-CPG, e.g., the average attack success rate of the proposed Adv-CPG is 28.1% and 2.86% higher compared to the SOTA noise-based attack methods and unconstrained attack methods, respectively.



## **5. A Grey-box Text Attack Framework using Explainable AI**

cs.CL

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2503.08226v1) [paper-pdf](http://arxiv.org/pdf/2503.08226v1)

**Authors**: Esther Chiramal, Kelvin Soh Boon Kai

**Abstract**: Explainable AI is a strong strategy implemented to understand complex black-box model predictions in a human interpretable language. It provides the evidence required to execute the use of trustworthy and reliable AI systems. On the other hand, however, it also opens the door to locating possible vulnerabilities in an AI model. Traditional adversarial text attack uses word substitution, data augmentation techniques and gradient-based attacks on powerful pre-trained Bidirectional Encoder Representations from Transformers (BERT) variants to generate adversarial sentences. These attacks are generally whitebox in nature and not practical as they can be easily detected by humans E.g. Changing the word from "Poor" to "Rich". We proposed a simple yet effective Grey-box cum Black-box approach that does not require the knowledge of the model while using a set of surrogate Transformer/BERT models to perform the attack using Explainable AI techniques. As Transformers are the current state-of-the-art models for almost all Natural Language Processing (NLP) tasks, an attack generated from BERT1 is transferable to BERT2. This transferability is made possible due to the attention mechanism in the transformer that allows the model to capture long-range dependencies in a sequence. Using the power of BERT generalisation via attention, we attempt to exploit how transformers learn by attacking a few surrogate transformer variants which are all based on a different architecture. We demonstrate that this approach is highly effective to generate semantically good sentences by changing as little as one word that is not detectable by humans while still fooling other BERT models.



## **6. Guardians of the Agentic System: Preventing Many Shots Jailbreak with Agentic System**

cs.CR

18 pages, 7 figures

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2502.16750v3) [paper-pdf](http://arxiv.org/pdf/2502.16750v3)

**Authors**: Saikat Barua, Mostafizur Rahman, Md Jafor Sadek, Rafiul Islam, Shehenaz Khaled, Ahmedul Kabir

**Abstract**: The autonomous AI agents using large language models can create undeniable values in all span of the society but they face security threats from adversaries that warrants immediate protective solutions because trust and safety issues arise. Considering the many-shot jailbreaking and deceptive alignment as some of the main advanced attacks, that cannot be mitigated by the static guardrails used during the supervised training, points out a crucial research priority for real world robustness. The combination of static guardrails in dynamic multi-agent system fails to defend against those attacks. We intend to enhance security for LLM-based agents through the development of new evaluation frameworks which identify and counter threats for safe operational deployment. Our work uses three examination methods to detect rogue agents through a Reverse Turing Test and analyze deceptive alignment through multi-agent simulations and develops an anti-jailbreaking system by testing it with GEMINI 1.5 pro and llama-3.3-70B, deepseek r1 models using tool-mediated adversarial scenarios. The detection capabilities are strong such as 94\% accuracy for GEMINI 1.5 pro yet the system suffers persistent vulnerabilities when under long attacks as prompt length increases attack success rates (ASR) and diversity metrics become ineffective in prediction while revealing multiple complex system faults. The findings demonstrate the necessity of adopting flexible security systems based on active monitoring that can be performed by the agents themselves together with adaptable interventions by system admin as the current models can create vulnerabilities that can lead to the unreliable and vulnerable system. So, in our work, we try to address such situations and propose a comprehensive framework to counteract the security issues.



## **7. Dialogue Injection Attack: Jailbreaking LLMs through Context Manipulation**

cs.CL

17 pages, 10 figures

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2503.08195v1) [paper-pdf](http://arxiv.org/pdf/2503.08195v1)

**Authors**: Wenlong Meng, Fan Zhang, Wendao Yao, Zhenyuan Guo, Yuwei Li, Chengkun Wei, Wenzhi Chen

**Abstract**: Large language models (LLMs) have demonstrated significant utility in a wide range of applications; however, their deployment is plagued by security vulnerabilities, notably jailbreak attacks. These attacks manipulate LLMs to generate harmful or unethical content by crafting adversarial prompts. While much of the current research on jailbreak attacks has focused on single-turn interactions, it has largely overlooked the impact of historical dialogues on model behavior. In this paper, we introduce a novel jailbreak paradigm, Dialogue Injection Attack (DIA), which leverages the dialogue history to enhance the success rates of such attacks. DIA operates in a black-box setting, requiring only access to the chat API or knowledge of the LLM's chat template. We propose two methods for constructing adversarial historical dialogues: one adapts gray-box prefilling attacks, and the other exploits deferred responses. Our experiments show that DIA achieves state-of-the-art attack success rates on recent LLMs, including Llama-3.1 and GPT-4o. Additionally, we demonstrate that DIA can bypass 5 different defense mechanisms, highlighting its robustness and effectiveness.



## **8. MAGIC: Mastering Physical Adversarial Generation in Context through Collaborative LLM Agents**

cs.CV

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2412.08014v2) [paper-pdf](http://arxiv.org/pdf/2412.08014v2)

**Authors**: Yun Xing, Nhat Chung, Jie Zhang, Yue Cao, Ivor Tsang, Yang Liu, Lei Ma, Qing Guo

**Abstract**: Physical adversarial attacks in driving scenarios can expose critical vulnerabilities in visual perception models. However, developing such attacks remains challenging due to diverse real-world environments and the requirement for maintaining visual naturality. Building upon this challenge, we reformulate physical adversarial attacks as a one-shot patch generation problem. Our approach generates adversarial patches through a deep generative model that considers the specific scene context, enabling direct physical deployment in matching environments. The primary challenge lies in simultaneously achieving two objectives: generating adversarial patches that effectively mislead object detection systems while determining contextually appropriate deployment within the scene. We propose MAGIC (Mastering Physical Adversarial Generation In Context), a novel framework powered by multi-modal LLM agents to address these challenges. MAGIC automatically understands scene context and generates adversarial patch through the synergistic interaction of language and vision capabilities. In particular, MAGIC orchestrates three specialized LLM agents: The adv-patch generation agent (GAgent) masters the creation of deceptive patches through strategic prompt engineering for text-to-image models. The adv-patch deployment agent (DAgent) ensures contextual coherence by determining optimal deployment strategies based on scene understanding. The self-examination agent (EAgent) completes this trilogy by providing critical oversight and iterative refinement of both processes. We validate our method on both digital and physical levels, i.e., nuImage and manually captured real-world scenes, where both statistical and visual results prove that our MAGIC is powerful and effective for attacking widely applied object detection systems, i.e., YOLO and DETR series.



## **9. MIGA: Mutual Information-Guided Attack on Denoising Models for Semantic Manipulation**

cs.CV

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2503.06966v2) [paper-pdf](http://arxiv.org/pdf/2503.06966v2)

**Authors**: Guanghao Li, Mingzhi Chen, Hao Yu, Shuting Dong, Wenhao Jiang, Ming Tang, Chun Yuan

**Abstract**: Deep learning-based denoising models have been widely employed in vision tasks, functioning as filters to eliminate noise while retaining crucial semantic information. Additionally, they play a vital role in defending against adversarial perturbations that threaten downstream tasks. However, these models can be intrinsically susceptible to adversarial attacks due to their dependence on specific noise assumptions. Existing attacks on denoising models mainly aim at deteriorating visual clarity while neglecting semantic manipulation, rendering them either easily detectable or limited in effectiveness. In this paper, we propose Mutual Information-Guided Attack (MIGA), the first method designed to directly attack deep denoising models by strategically disrupting their ability to preserve semantic content via adversarial perturbations. By minimizing the mutual information between the original and denoised images, a measure of semantic similarity. MIGA forces the denoiser to produce perceptually clean yet semantically altered outputs. While these images appear visually plausible, they encode systematically distorted semantics, revealing a fundamental vulnerability in denoising models. These distortions persist in denoised outputs and can be quantitatively assessed through downstream task performance. We propose new evaluation metrics and systematically assess MIGA on four denoising models across five datasets, demonstrating its consistent effectiveness in disrupting semantic fidelity. Our findings suggest that denoising models are not always robust and can introduce security risks in real-world applications.



## **10. Towards Million-Scale Adversarial Robustness Evaluation With Stronger Individual Attacks**

cs.LG

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2411.15210v4) [paper-pdf](http://arxiv.org/pdf/2411.15210v4)

**Authors**: Yong Xie, Weijie Zheng, Hanxun Huang, Guangnan Ye, Xingjun Ma

**Abstract**: As deep learning models are increasingly deployed in safety-critical applications, evaluating their vulnerabilities to adversarial perturbations is essential for ensuring their reliability and trustworthiness. Over the past decade, a large number of white-box adversarial robustness evaluation methods (i.e., attacks) have been proposed, ranging from single-step to multi-step methods and from individual to ensemble methods. Despite these advances, challenges remain in conducting meaningful and comprehensive robustness evaluations, particularly when it comes to large-scale testing and ensuring evaluations reflect real-world adversarial risks. In this work, we focus on image classification models and propose a novel individual attack method, Probability Margin Attack (PMA), which defines the adversarial margin in the probability space rather than the logits space. We analyze the relationship between PMA and existing cross-entropy or logits-margin-based attacks, and show that PMA can outperform the current state-of-the-art individual methods. Building on PMA, we propose two types of ensemble attacks that balance effectiveness and efficiency. Furthermore, we create a million-scale dataset, CC1M, derived from the existing CC3M dataset, and use it to conduct the first million-scale white-box adversarial robustness evaluation of adversarially-trained ImageNet models. Our findings provide valuable insights into the robustness gaps between individual versus ensemble attacks and small-scale versus million-scale evaluations.



## **11. Human-in-the-Loop Generation of Adversarial Texts: A Case Study on Tibetan Script**

cs.CL

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2412.12478v2) [paper-pdf](http://arxiv.org/pdf/2412.12478v2)

**Authors**: Xi Cao, Yuan Sun, Jiajun Li, Quzong Gesang, Nuo Qun, Tashi Nyima

**Abstract**: DNN-based language models perform excellently on various tasks, but even SOTA LLMs are susceptible to textual adversarial attacks. Adversarial texts play crucial roles in multiple subfields of NLP. However, current research has the following issues. (1) Most textual adversarial attack methods target rich-resourced languages. How do we generate adversarial texts for less-studied languages? (2) Most textual adversarial attack methods are prone to generating invalid or ambiguous adversarial texts. How do we construct high-quality adversarial robustness benchmarks? (3) New language models may be immune to part of previously generated adversarial texts. How do we update adversarial robustness benchmarks? To address the above issues, we introduce HITL-GAT, a system based on a general approach to human-in-the-loop generation of adversarial texts. HITL-GAT contains four stages in one pipeline: victim model construction, adversarial example generation, high-quality benchmark construction, and adversarial robustness evaluation. Additionally, we utilize HITL-GAT to make a case study on Tibetan script which can be a reference for the adversarial research of other less-studied languages.



## **12. Safety Guardrails for LLM-Enabled Robots**

cs.RO

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2503.07885v1) [paper-pdf](http://arxiv.org/pdf/2503.07885v1)

**Authors**: Zachary Ravichandran, Alexander Robey, Vijay Kumar, George J. Pappas, Hamed Hassani

**Abstract**: Although the integration of large language models (LLMs) into robotics has unlocked transformative capabilities, it has also introduced significant safety concerns, ranging from average-case LLM errors (e.g., hallucinations) to adversarial jailbreaking attacks, which can produce harmful robot behavior in real-world settings. Traditional robot safety approaches do not address the novel vulnerabilities of LLMs, and current LLM safety guardrails overlook the physical risks posed by robots operating in dynamic real-world environments. In this paper, we propose RoboGuard, a two-stage guardrail architecture to ensure the safety of LLM-enabled robots. RoboGuard first contextualizes pre-defined safety rules by grounding them in the robot's environment using a root-of-trust LLM, which employs chain-of-thought (CoT) reasoning to generate rigorous safety specifications, such as temporal logic constraints. RoboGuard then resolves potential conflicts between these contextual safety specifications and a possibly unsafe plan using temporal logic control synthesis, which ensures safety compliance while minimally violating user preferences. Through extensive simulation and real-world experiments that consider worst-case jailbreaking attacks, we demonstrate that RoboGuard reduces the execution of unsafe plans from 92% to below 2.5% without compromising performance on safe plans. We also demonstrate that RoboGuard is resource-efficient, robust against adaptive attacks, and significantly enhanced by enabling its root-of-trust LLM to perform CoT reasoning. These results underscore the potential of RoboGuard to mitigate the safety risks and enhance the reliability of LLM-enabled robots.



## **13. ReLATE: Resilient Learner Selection for Multivariate Time-Series Classification Against Adversarial Attacks**

cs.LG

Accepted by the AAAI-25 Workshop on Artificial Intelligence for Time  Series Analysis (AI4TS)

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2503.07882v1) [paper-pdf](http://arxiv.org/pdf/2503.07882v1)

**Authors**: Cagla Ipek Kocal, Onat Gungor, Aaron Tartz, Tajana Rosing, Baris Aksanli

**Abstract**: Minimizing computational overhead in time-series classification, particularly in deep learning models, presents a significant challenge. This challenge is further compounded by adversarial attacks, emphasizing the need for resilient methods that ensure robust performance and efficient model selection. We introduce ReLATE, a framework that identifies robust learners based on dataset similarity, reduces computational overhead, and enhances resilience. ReLATE maintains multiple deep learning models in well-known adversarial attack scenarios, capturing model performance. ReLATE identifies the most analogous dataset to a given target using a similarity metric, then applies the optimal model from the most similar dataset. ReLATE reduces computational overhead by an average of 81.2%, enhancing adversarial resilience and streamlining robust model selection, all without sacrificing performance, within 4.2% of Oracle.



## **14. On the Byzantine Fault Tolerance of signSGD with Majority Vote**

cs.LG

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2502.19170v2) [paper-pdf](http://arxiv.org/pdf/2502.19170v2)

**Authors**: Emanuele Mengoli, Luzius Moll, Virgilio Strozzi, El-Mahdi El-Mhamdi

**Abstract**: In distributed learning, sign-based compression algorithms such as signSGD with majority vote provide a lightweight alternative to SGD with an additional advantage: fault tolerance (almost) for free. However, for signSGD with majority vote, this fault tolerance has been shown to cover only the case of weaker adversaries, i.e., ones that are not omniscient or cannot collude to base their attack on common knowledge and strategy. In this work, we close this gap and provide new insights into how signSGD with majority vote can be resilient against omniscient and colluding adversaries, which craft an attack after communicating with other adversaries, thus having better information to perform the most damaging attack based on a common optimal strategy. Our core contribution is in providing a proof that begins by defining the omniscience framework and the strongest possible damage against signSGD with majority vote without imposing any restrictions on the attacker. Thanks to the filtering effect of the sign-based method, we upper-bound the space of attacks to the optimal strategy for maximizing damage by an attacker. Hence, we derive an explicit probabilistic bound in terms of incorrect aggregation without resorting to unknown constants, providing a convergence bound on signSGD with majority vote in the presence of Byzantine attackers, along with a precise convergence rate. Our findings are supported by experiments on the MNIST dataset in a distributed learning environment with adversaries of varying strength.



## **15. Runtime Detection of Adversarial Attacks in AI Accelerators Using Performance Counters**

cs.CR

7 pages, 8 figures

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2503.07568v1) [paper-pdf](http://arxiv.org/pdf/2503.07568v1)

**Authors**: Habibur Rahaman, Atri Chatterjee, Swarup Bhunia

**Abstract**: Rapid adoption of AI technologies raises several major security concerns, including the risks of adversarial perturbations, which threaten the confidentiality and integrity of AI applications. Protecting AI hardware from misuse and diverse security threats is a challenging task. To address this challenge, we propose SAMURAI, a novel framework for safeguarding against malicious usage of AI hardware and its resilience to attacks. SAMURAI introduces an AI Performance Counter (APC) for tracking dynamic behavior of an AI model coupled with an on-chip Machine Learning (ML) analysis engine, known as TANTO (Trained Anomaly Inspection Through Trace Observation). APC records the runtime profile of the low-level hardware events of different AI operations. Subsequently, the summary information recorded by the APC is processed by TANTO to efficiently identify potential security breaches and ensure secure, responsible use of AI. SAMURAI enables real-time detection of security threats and misuse without relying on traditional software-based solutions that require model integration. Experimental results demonstrate that SAMURAI achieves up to 97% accuracy in detecting adversarial attacks with moderate overhead on various AI models, significantly outperforming conventional software-based approaches. It enhances security and regulatory compliance, providing a comprehensive solution for safeguarding AI against emergent threats.



## **16. Transform-Dependent Adversarial Attacks**

cs.CV

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2406.08443v2) [paper-pdf](http://arxiv.org/pdf/2406.08443v2)

**Authors**: Yaoteng Tan, Zikui Cai, M. Salman Asif

**Abstract**: Deep networks are highly vulnerable to adversarial attacks, yet conventional attack methods utilize static adversarial perturbations that induce fixed mispredictions. In this work, we exploit an overlooked property of adversarial perturbations--their dependence on image transforms--and introduce transform-dependent adversarial attacks. Unlike traditional attacks, our perturbations exhibit metamorphic properties, enabling diverse adversarial effects as a function of transformation parameters. We demonstrate that this transform-dependent vulnerability exists across different architectures (e.g., CNN and transformer), vision tasks (e.g., image classification and object detection), and a wide range of image transforms. Additionally, we show that transform-dependent perturbations can serve as a defense mechanism, preventing sensitive information disclosure when image enhancement transforms pose a risk of revealing private content. Through analysis in blackbox and defended model settings, we show that transform-dependent perturbations achieve high targeted attack success rates, outperforming state-of-the-art transfer attacks by 17-31% in blackbox scenarios. Our work introduces novel, controllable paradigm for adversarial attack deployment, revealing a previously overlooked vulnerability in deep networks.



## **17. Learning to Localize Leakage of Cryptographic Sensitive Variables**

cs.LG

52 pages, 30 figures. Our code can be found at  https://github.com/jimgammell/learning_to_localize_leakage

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2503.07464v1) [paper-pdf](http://arxiv.org/pdf/2503.07464v1)

**Authors**: Jimmy Gammell, Anand Raghunathan, Abolfazl Hashemi, Kaushik Roy

**Abstract**: While cryptographic algorithms such as the ubiquitous Advanced Encryption Standard (AES) are secure, *physical implementations* of these algorithms in hardware inevitably 'leak' sensitive data such as cryptographic keys. A particularly insidious form of leakage arises from the fact that hardware consumes power and emits radiation in a manner that is statistically associated with the data it processes and the instructions it executes. Supervised deep learning has emerged as a state-of-the-art tool for carrying out *side-channel attacks*, which exploit this leakage by learning to map power/radiation measurements throughout encryption to the sensitive data operated on during that encryption. In this work we develop a principled deep learning framework for determining the relative leakage due to measurements recorded at different points in time, in order to inform *defense* against such attacks. This information is invaluable to cryptographic hardware designers for understanding *why* their hardware leaks and how they can mitigate it (e.g. by indicating the particular sections of code or electronic components which are responsible). Our framework is based on an adversarial game between a family of classifiers trained to estimate the conditional distributions of sensitive data given subsets of measurements, and a budget-constrained noise distribution which probabilistically erases individual measurements to maximize the loss of these classifiers. We demonstrate our method's efficacy and ability to overcome limitations of prior work through extensive experimental comparison with 8 baseline methods using 3 evaluation metrics and 6 publicly-available power/EM trace datasets from AES, ECC and RSA implementations. We provide an open-source PyTorch implementation of these experiments.



## **18. The Uncanny Valley: Exploring Adversarial Robustness from a Flatness Perspective**

cs.LG

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2405.16918v2) [paper-pdf](http://arxiv.org/pdf/2405.16918v2)

**Authors**: Nils Philipp Walter, Linara Adilova, Jilles Vreeken, Michael Kamp

**Abstract**: Flatness of the loss surface not only correlates positively with generalization, but is also related to adversarial robustness since perturbations of inputs relate non-linearly to perturbations of weights. In this paper, we empirically analyze the relation between adversarial examples and relative flatness with respect to the parameters of one layer. We observe a peculiar property of adversarial examples in the context of relative flatness: during an iterative first-order white-box attack, the flatness of the loss surface measured around the adversarial example first becomes sharper until the label is flipped, but if we keep the attack running, it runs into a flat uncanny valley where the label remains flipped. In extensive experiments, we observe this phenomenon across various model architectures and datasets, even for adversarially trained models. Our results also extend to large language models (LLMs), but due to the discrete nature of the input space and comparatively weak attacks, adversarial examples rarely reach truly flat regions. Most importantly, this phenomenon shows that flatness alone cannot explain adversarial robustness unless we can also guarantee the behavior of the function around the examples. We, therefore theoretically connect relative flatness to adversarial robustness by bounding the third derivative of the loss surface, underlining the need for flatness in combination with a low global Lipschitz constant for a robust model.



## **19. MIBench: A Comprehensive Framework for Benchmarking Model Inversion Attack and Defense**

cs.CV

20 pages

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2410.05159v3) [paper-pdf](http://arxiv.org/pdf/2410.05159v3)

**Authors**: Yixiang Qiu, Hongyao Yu, Hao Fang, Tianqu Zhuang, Wenbo Yu, Bin Chen, Xuan Wang, Shu-Tao Xia, Ke Xu

**Abstract**: Model Inversion (MI) attacks aim at leveraging the output information of target models to reconstruct privacy-sensitive training data, raising critical concerns regarding the privacy vulnerabilities of Deep Neural Networks (DNNs). Unfortunately, in tandem with the rapid evolution of MI attacks, the absence of a comprehensive benchmark with standardized metrics and reproducible implementations has emerged as a formidable challenge. This deficiency has hindered objective comparison of methodological advancements and reliable assessment of defense efficacy. To address this critical gap, we build the first practical benchmark named MIBench for systematic evaluation of model inversion attacks and defenses. This benchmark bases on an extensible and reproducible modular-based toolbox which currently integrates a total of 19 state-of-the-art attack and defense methods and encompasses 9 standardized evaluation protocols. Capitalizing on this foundation, we conduct extensive evaluation from multiple perspectives to holistically compare and analyze various methods across different scenarios, such as the impact of target resolution, model predictive power, defense performance and adversarial robustness.



## **20. State Frequency Estimation for Anomaly Detection**

cs.LG

12 pages

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2412.03442v2) [paper-pdf](http://arxiv.org/pdf/2412.03442v2)

**Authors**: Clinton Cao, Agathe Blaise, Annibale Panichella, Sicco Verwer

**Abstract**: Many works have studied the efficacy of state machines for detecting anomalies within NetFlows. These works typically learn a model from unlabeled data and compute anomaly scores for arbitrary traces based on their likelihood of occurrence or how well they fit within the model. However, these methods do not dynamically adapt their scores based on the traces seen at test time. This becomes a problem when an adversary produces seemingly common traces in their attack, causing the model to miss the detection by assigning low anomaly scores. We propose SEQUENT, a new unsupervised approach that uses the state visit frequency of a state machine to adapt its scoring dynamically for anomaly detection. SEQUENT subsequently uses the scores to generate root causes for anomalies. These allow the grouping of alarms and simplify the analysis of anomalies. We evaluate SEQUENT's effectiveness in detecting network anomalies on three publicly available NetFlow datasets and compare its performance against various existing unsupervised anomaly detection methods. Our evaluation shows promising results for using the state visit frequency of a state machine to detect network anomalies.



## **21. Machine Against the RAG: Jamming Retrieval-Augmented Generation with Blocker Documents**

cs.CR

To appear in USENIX Security Symposium 2025

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2406.05870v4) [paper-pdf](http://arxiv.org/pdf/2406.05870v4)

**Authors**: Avital Shafran, Roei Schuster, Vitaly Shmatikov

**Abstract**: Retrieval-augmented generation (RAG) systems respond to queries by retrieving relevant documents from a knowledge database and applying an LLM to the retrieved documents. We demonstrate that RAG systems that operate on databases with untrusted content are vulnerable to denial-of-service attacks we call jamming. An adversary can add a single ``blocker'' document to the database that will be retrieved in response to a specific query and result in the RAG system not answering this query, ostensibly because it lacks relevant information or because the answer is unsafe.   We describe and measure the efficacy of several methods for generating blocker documents, including a new method based on black-box optimization. Our method (1) does not rely on instruction injection, (2) does not require the adversary to know the embedding or LLM used by the target RAG system, and (3) does not employ an auxiliary LLM.   We evaluate jamming attacks on several embeddings and LLMs and demonstrate that the existing safety metrics for LLMs do not capture their vulnerability to jamming. We then discuss defenses against blocker documents.



## **22. PGD-Imp: Rethinking and Unleashing Potential of Classic PGD with Dual Strategies for Imperceptible Adversarial Attacks**

cs.LG

Accepted by IEEE ICASSP 2025. Please cite this paper using the  following format: J. Li, Z. Yu, Z. He, Z. Wang, X. Kang, "PGD-Imp: Rethinking  and Unleashing Potential of Classic PGD with Dual Strategies for  Imperceptible Adversarial Attacks," in proc. of International Conference on  Acoustics, Speech, and Signal Processing 2025 (ICASSP 2025), Hyderabad,  India, 2025-4-06 to 2025-04-11

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2412.11168v3) [paper-pdf](http://arxiv.org/pdf/2412.11168v3)

**Authors**: Jin Li, Zitong Yu, Ziqiang He, Z. Jane Wang, Xiangui Kang

**Abstract**: Imperceptible adversarial attacks have recently attracted increasing research interests. Existing methods typically incorporate external modules or loss terms other than a simple $l_p$-norm into the attack process to achieve imperceptibility, while we argue that such additional designs may not be necessary. In this paper, we rethink the essence of imperceptible attacks and propose two simple yet effective strategies to unleash the potential of PGD, the common and classical attack, for imperceptibility from an optimization perspective. Specifically, the Dynamic Step Size is introduced to find the optimal solution with minimal attack cost towards the decision boundary of the attacked model, and the Adaptive Early Stop strategy is adopted to reduce the redundant strength of adversarial perturbations to the minimum level. The proposed PGD-Imperceptible (PGD-Imp) attack achieves state-of-the-art results in imperceptible adversarial attacks for both untargeted and targeted scenarios. When performing untargeted attacks against ResNet-50, PGD-Imp attains 100$\%$ (+0.3$\%$) ASR, 0.89 (-1.76) $l_2$ distance, and 52.93 (+9.2) PSNR with 57s (-371s) running time, significantly outperforming existing methods.



## **23. Breaking the Limits of Quantization-Aware Defenses: QADT-R for Robustness Against Patch-Based Adversarial Attacks in QNNs**

cs.CV

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2503.07058v1) [paper-pdf](http://arxiv.org/pdf/2503.07058v1)

**Authors**: Amira Guesmi, Bassem Ouni, Muhammad Shafique

**Abstract**: Quantized Neural Networks (QNNs) have emerged as a promising solution for reducing model size and computational costs, making them well-suited for deployment in edge and resource-constrained environments. While quantization is known to disrupt gradient propagation and enhance robustness against pixel-level adversarial attacks, its effectiveness against patch-based adversarial attacks remains largely unexplored. In this work, we demonstrate that adversarial patches remain highly transferable across quantized models, achieving over 70\% attack success rates (ASR) even at extreme bit-width reductions (e.g., 2-bit). This challenges the common assumption that quantization inherently mitigates adversarial threats. To address this, we propose Quantization-Aware Defense Training with Randomization (QADT-R), a novel defense strategy that integrates Adaptive Quantization-Aware Patch Generation (A-QAPA), Dynamic Bit-Width Training (DBWT), and Gradient-Inconsistent Regularization (GIR) to enhance resilience against highly transferable patch-based attacks. A-QAPA generates adversarial patches within quantized models, ensuring robustness across different bit-widths. DBWT introduces bit-width cycling during training to prevent overfitting to a specific quantization setting, while GIR injects controlled gradient perturbations to disrupt adversarial optimization. Extensive evaluations on CIFAR-10 and ImageNet show that QADT-R reduces ASR by up to 25\% compared to prior defenses such as PBAT and DWQ. Our findings further reveal that PBAT-trained models, while effective against seen patch configurations, fail to generalize to unseen patches due to quantization shift. Additionally, our empirical analysis of gradient alignment, spatial sensitivity, and patch visibility provides insights into the mechanisms that contribute to the high transferability of patch-based attacks in QNNs.



## **24. Utilizing Jailbreak Probability to Attack and Safeguard Multimodal LLMs**

cs.CR

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2503.06989v1) [paper-pdf](http://arxiv.org/pdf/2503.06989v1)

**Authors**: Wenzhuo Xu, Zhipeng Wei, Xiongtao Sun, Deyue Zhang, Dongdong Yang, Quanchen Zou, Xiangzheng Zhang

**Abstract**: Recently, Multimodal Large Language Models (MLLMs) have demonstrated their superior ability in understanding multimodal contents. However, they remain vulnerable to jailbreak attacks, which exploit weaknesses in their safety alignment to generate harmful responses. Previous studies categorize jailbreaks as successful or failed based on whether responses contain malicious content. However, given the stochastic nature of MLLM responses, this binary classification of an input's ability to jailbreak MLLMs is inappropriate. Derived from this viewpoint, we introduce jailbreak probability to quantify the jailbreak potential of an input, which represents the likelihood that MLLMs generated a malicious response when prompted with this input. We approximate this probability through multiple queries to MLLMs. After modeling the relationship between input hidden states and their corresponding jailbreak probability using Jailbreak Probability Prediction Network (JPPN), we use continuous jailbreak probability for optimization. Specifically, we propose Jailbreak-Probability-based Attack (JPA) that optimizes adversarial perturbations on inputs to maximize jailbreak probability. To counteract attacks, we also propose two defensive methods: Jailbreak-Probability-based Finetuning (JPF) and Jailbreak-Probability-based Defensive Noise (JPDN), which minimizes jailbreak probability in the MLLM parameters and input space, respectively. Extensive experiments show that (1) JPA yields improvements (up to 28.38\%) under both white and black box settings compared to previous methods with small perturbation bounds and few iterations. (2) JPF and JPDN significantly reduce jailbreaks by at most over 60\%. Both of the above results demonstrate the significance of introducing jailbreak probability to make nuanced distinctions among input jailbreak abilities.



## **25. Vulnerabilities in AI-generated Image Detection: The Challenge of Adversarial Attacks**

cs.CV

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2407.20836v3) [paper-pdf](http://arxiv.org/pdf/2407.20836v3)

**Authors**: Yunfeng Diao, Naixin Zhai, Changtao Miao, Zitong Yu, Xingxing Wei, Xun Yang, Meng Wang

**Abstract**: Recent advancements in image synthesis, particularly with the advent of GAN and Diffusion models, have amplified public concerns regarding the dissemination of disinformation. To address such concerns, numerous AI-generated Image (AIGI) Detectors have been proposed and achieved promising performance in identifying fake images. However, there still lacks a systematic understanding of the adversarial robustness of AIGI detectors. In this paper, we examine the vulnerability of state-of-the-art AIGI detectors against adversarial attack under white-box and black-box settings, which has been rarely investigated so far. To this end, we propose a new method to attack AIGI detectors. First, inspired by the obvious difference between real images and fake images in the frequency domain, we add perturbations under the frequency domain to push the image away from its original frequency distribution. Second, we explore the full posterior distribution of the surrogate model to further narrow this gap between heterogeneous AIGI detectors, e.g. transferring adversarial examples across CNNs and ViTs. This is achieved by introducing a novel post-train Bayesian strategy that turns a single surrogate into a Bayesian one, capable of simulating diverse victim models using one pre-trained surrogate, without the need for re-training. We name our method as Frequency-based Post-train Bayesian Attack, or FPBA. Through FPBA, we show that adversarial attack is truly a real threat to AIGI detectors, because FPBA can deliver successful black-box attacks across models, generators, defense methods, and even evade cross-generator detection, which is a crucial real-world detection scenario. The code will be shared upon acceptance.



## **26. CtrlRAG: Black-box Adversarial Attacks Based on Masked Language Models in Retrieval-Augmented Language Generation**

cs.CL

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2503.06950v1) [paper-pdf](http://arxiv.org/pdf/2503.06950v1)

**Authors**: Runqi Sui

**Abstract**: Retrieval-Augmented Generation (RAG) systems enhance Large Language Models (LLMs) by integrating external knowledge bases. However, this integration introduces a new security threat: adversaries can exploit the retrieval mechanism to inject malicious content into the knowledge base, thereby influencing the generated responses. Based on this attack vector, we propose CtrlRAG, a novel attack method designed for RAG system in the black-box setting, which aligns with real-world scenarios. Unlike existing attack methods, CtrlRAG introduces a perturbation mechanism using Masked Language Model (MLM) to dynamically optimize malicious content in response to changes in the retrieved context. Experimental results demonstrate that CtrlRAG outperforms three baseline methods in both Emotional Manipulation and Hallucination Amplification objectives. Furthermore, we evaluate three existing defense mechanisms, revealing their limited effectiveness against CtrlRAG and underscoring the urgent need for more robust defenses.



## **27. When Lighting Deceives: Exposing Vision-Language Models' Illumination Vulnerability Through Illumination Transformation Attack**

cs.CV

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2503.06903v1) [paper-pdf](http://arxiv.org/pdf/2503.06903v1)

**Authors**: Hanqing Liu, Shouwei Ruan, Yao Huang, Shiji Zhao, Xingxing Wei

**Abstract**: Vision-Language Models (VLMs) have achieved remarkable success in various tasks, yet their robustness to real-world illumination variations remains largely unexplored. To bridge this gap, we propose \textbf{I}llumination \textbf{T}ransformation \textbf{A}ttack (\textbf{ITA}), the first framework to systematically assess VLMs' robustness against illumination changes. However, there still exist two key challenges: (1) how to model global illumination with fine-grained control to achieve diverse lighting conditions and (2) how to ensure adversarial effectiveness while maintaining naturalness. To address the first challenge, we innovatively decompose global illumination into multiple parameterized point light sources based on the illumination rendering equation. This design enables us to model more diverse lighting variations that previous methods could not capture. Then, by integrating these parameterized lighting variations with physics-based lighting reconstruction techniques, we could precisely render such light interactions in the original scenes, finally meeting the goal of fine-grained lighting control. For the second challenge, by controlling illumination through the lighting reconstrution model's latent space rather than direct pixel manipulation, we inherently preserve physical lighting priors. Furthermore, to prevent potential reconstruction artifacts, we design additional perceptual constraints for maintaining visual consistency with original images and diversity constraints for avoiding light source convergence.   Extensive experiments demonstrate that our ITA could significantly reduce the performance of advanced VLMs, e.g., LLaVA-1.6, while possessing competitive naturalness, exposing VLMS' critical illuminiation vulnerabilities.



## **28. Exploring the Adversarial Vulnerabilities of Vision-Language-Action Models in Robotics**

cs.RO

Github: https://github.com/William-wAng618/roboticAttack Homepage:  https://vlaattacker.github.io/

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2411.13587v3) [paper-pdf](http://arxiv.org/pdf/2411.13587v3)

**Authors**: Taowen Wang, Cheng Han, James Chenhao Liang, Wenhao Yang, Dongfang Liu, Luna Xinyu Zhang, Qifan Wang, Jiebo Luo, Ruixiang Tang

**Abstract**: Recently in robotics, Vision-Language-Action (VLA) models have emerged as a transformative approach, enabling robots to execute complex tasks by integrating visual and linguistic inputs within an end-to-end learning framework. While VLA models offer significant capabilities, they also introduce new attack surfaces, making them vulnerable to adversarial attacks. With these vulnerabilities largely unexplored, this paper systematically quantifies the robustness of VLA-based robotic systems. Recognizing the unique demands of robotic execution, our attack objectives target the inherent spatial and functional characteristics of robotic systems. In particular, we introduce two untargeted attack objectives that leverage spatial foundations to destabilize robotic actions, and a targeted attack objective that manipulates the robotic trajectory. Additionally, we design an adversarial patch generation approach that places a small, colorful patch within the camera's view, effectively executing the attack in both digital and physical environments. Our evaluation reveals a marked degradation in task success rates, with up to a 100\% reduction across a suite of simulated robotic tasks, highlighting critical security gaps in current VLA architectures. By unveiling these vulnerabilities and proposing actionable evaluation metrics, we advance both the understanding and enhancement of safety for VLA-based robotic systems, underscoring the necessity for continuously developing robust defense strategies prior to physical-world deployments.



## **29. Quantum Chernoff divergence in advantage distillation for quantum key distribution and device-independent quantum key distribution**

quant-ph

Close to published version

**SubmitDate**: 2025-03-09    [abs](http://arxiv.org/abs/2212.06975v3) [paper-pdf](http://arxiv.org/pdf/2212.06975v3)

**Authors**: Mikka Stasiuk, Norbert Lütkenhaus, Ernest Y. -Z. Tan

**Abstract**: Device-independent quantum key distribution (DIQKD) aims to mitigate adversarial exploitation of imperfections in quantum devices, by providing an approach for secret key distillation with modest security assumptions. Advantage distillation, a two-way communication procedure in error correction, has proven effective in raising noise tolerances in both device-dependent and device-independent QKD. Previously, device-independent security proofs against IID collective attacks were developed for an advantage distillation protocol known as the repetition-code protocol, based on security conditions involving the fidelity between some states in the protocol. However, there exists a gap between the sufficient and necessary security conditions, which hinders the calculation of tight noise-tolerance bounds based on the fidelity. We close this gap by presenting an alternative proof structure that replaces the fidelity with the quantum Chernoff divergence, a distinguishability measure that arises in symmetric hypothesis testing. Working in the IID collective attacks model, we derive matching sufficient and necessary conditions for the repetition-code protocol to be secure (up to a natural conjecture regarding the latter case) in terms of the quantum Chernoff divergence, hence indicating that this serves as the relevant quantity of interest for this protocol. Furthermore, using this security condition we obtain some improvements over previous results on the noise tolerance thresholds for DIQKD. Our results provide insight into a fundamental question in quantum information theory regarding the circumstances under which DIQKD is possible.



## **30. AnywhereDoor: Multi-Target Backdoor Attacks on Object Detection**

cs.CR

**SubmitDate**: 2025-03-09    [abs](http://arxiv.org/abs/2503.06529v1) [paper-pdf](http://arxiv.org/pdf/2503.06529v1)

**Authors**: Jialin Lu, Junjie Shan, Ziqi Zhao, Ka-Ho Chow

**Abstract**: As object detection becomes integral to many safety-critical applications, understanding its vulnerabilities is essential. Backdoor attacks, in particular, pose a serious threat by implanting hidden triggers in victim models, which adversaries can later exploit to induce malicious behaviors during inference. However, current understanding is limited to single-target attacks, where adversaries must define a fixed malicious behavior (target) before training, making inference-time adaptability impossible. Given the large output space of object detection (including object existence prediction, bounding box estimation, and classification), the feasibility of flexible, inference-time model control remains unexplored. This paper introduces AnywhereDoor, a multi-target backdoor attack for object detection. Once implanted, AnywhereDoor allows adversaries to make objects disappear, fabricate new ones, or mislabel them, either across all object classes or specific ones, offering an unprecedented degree of control. This flexibility is enabled by three key innovations: (i) objective disentanglement to scale the number of supported targets; (ii) trigger mosaicking to ensure robustness even against region-based detectors; and (iii) strategic batching to address object-level data imbalances that hinder manipulation. Extensive experiments demonstrate that AnywhereDoor grants attackers a high degree of control, improving attack success rates by 26% compared to adaptations of existing methods for such flexible control.



## **31. Visual Privacy Auditing with Diffusion Models**

cs.LG

Published in Transactions on Machine Learning Research (TMLR)

**SubmitDate**: 2025-03-09    [abs](http://arxiv.org/abs/2403.07588v2) [paper-pdf](http://arxiv.org/pdf/2403.07588v2)

**Authors**: Kristian Schwethelm, Johannes Kaiser, Moritz Knolle, Sarah Lockfisch, Daniel Rueckert, Alexander Ziller

**Abstract**: Data reconstruction attacks on machine learning models pose a substantial threat to privacy, potentially leaking sensitive information. Although defending against such attacks using differential privacy (DP) provides theoretical guarantees, determining appropriate DP parameters remains challenging. Current formal guarantees on the success of data reconstruction suffer from overly stringent assumptions regarding adversary knowledge about the target data, particularly in the image domain, raising questions about their real-world applicability. In this work, we empirically investigate this discrepancy by introducing a reconstruction attack based on diffusion models (DMs) that only assumes adversary access to real-world image priors and specifically targets the DP defense. We find that (1) real-world data priors significantly influence reconstruction success, (2) current reconstruction bounds do not model the risk posed by data priors well, and (3) DMs can serve as heuristic auditing tools for visualizing privacy leakage.



## **32. One Perturbation is Enough: On Generating Universal Adversarial Perturbations against Vision-Language Pre-training Models**

cs.CV

**SubmitDate**: 2025-03-09    [abs](http://arxiv.org/abs/2406.05491v3) [paper-pdf](http://arxiv.org/pdf/2406.05491v3)

**Authors**: Hao Fang, Jiawei Kong, Wenbo Yu, Bin Chen, Jiawei Li, Hao Wu, Shutao Xia, Ke Xu

**Abstract**: Vision-Language Pre-training (VLP) models have exhibited unprecedented capability in many applications by taking full advantage of the multimodal alignment. However, previous studies have shown they are vulnerable to maliciously crafted adversarial samples. Despite recent success, these methods are generally instance-specific and require generating perturbations for each input sample. In this paper, we reveal that VLP models are also vulnerable to the instance-agnostic universal adversarial perturbation (UAP). Specifically, we design a novel Contrastive-training Perturbation Generator with Cross-modal conditions (C-PGC) to achieve the attack. In light that the pivotal multimodal alignment is achieved through the advanced contrastive learning technique, we devise to turn this powerful weapon against themselves, i.e., employ a malicious version of contrastive learning to train the C-PGC based on our carefully crafted positive and negative image-text pairs for essentially destroying the alignment relationship learned by VLP models. Besides, C-PGC fully utilizes the characteristics of Vision-and-Language (V+L) scenarios by incorporating both unimodal and cross-modal information as effective guidance. Extensive experiments show that C-PGC successfully forces adversarial samples to move away from their original area in the VLP model's feature space, thus essentially enhancing attacks across various victim models and V+L tasks. The GitHub repository is available at https://github.com/ffhibnese/CPGC_VLP_Universal_Attacks.



## **33. Long-tailed Adversarial Training with Self-Distillation**

cs.CV

ICLR 2025

**SubmitDate**: 2025-03-09    [abs](http://arxiv.org/abs/2503.06461v1) [paper-pdf](http://arxiv.org/pdf/2503.06461v1)

**Authors**: Seungju Cho, Hongsin Lee, Changick Kim

**Abstract**: Adversarial training significantly enhances adversarial robustness, yet superior performance is predominantly achieved on balanced datasets.   Addressing adversarial robustness in the context of unbalanced or long-tailed distributions is considerably more challenging, mainly due to the scarcity of tail data instances.   Previous research on adversarial robustness within long-tailed distributions has primarily focused on combining traditional long-tailed natural training with existing adversarial robustness methods.   In this study, we provide an in-depth analysis for the challenge that adversarial training struggles to achieve high performance on tail classes in long-tailed distributions.   Furthermore, we propose a simple yet effective solution to advance adversarial robustness on long-tailed distributions through a novel self-distillation technique.   Specifically, this approach leverages a balanced self-teacher model, which is trained using a balanced dataset sampled from the original long-tailed dataset. Our extensive experiments demonstrate state-of-the-art performance in both clean and robust accuracy for long-tailed adversarial robustness, with significant improvements in tail class performance on various datasets. We improve the accuracy against PGD attacks for tail classes by 20.3, 7.1, and 3.8 percentage points on CIFAR-10, CIFAR-100, and Tiny-ImageNet, respectively, while achieving the highest robust accuracy.



## **34. Adversarial Robustness of Discriminative Self-Supervised Learning in Vision**

cs.CV

53 pages

**SubmitDate**: 2025-03-08    [abs](http://arxiv.org/abs/2503.06361v1) [paper-pdf](http://arxiv.org/pdf/2503.06361v1)

**Authors**: Ömer Veysel Çağatan, Ömer Faruk Tal, M. Emre Gürsoy

**Abstract**: Self-supervised learning (SSL) has advanced significantly in visual representation learning, yet comprehensive evaluations of its adversarial robustness remain limited. In this study, we evaluate the adversarial robustness of seven discriminative self-supervised models and one supervised model across diverse tasks, including ImageNet classification, transfer learning, segmentation, and detection. Our findings suggest that discriminative SSL models generally exhibit better robustness to adversarial attacks compared to their supervised counterpart on ImageNet, with this advantage extending to transfer learning when using linear evaluation. However, when fine-tuning is applied, the robustness gap between SSL and supervised models narrows considerably. Similarly, this robustness advantage diminishes in segmentation and detection tasks. We also investigate how various factors might influence adversarial robustness, including architectural choices, training duration, data augmentations, and batch sizes. Our analysis contributes to the ongoing exploration of adversarial robustness in visual self-supervised representation systems.



## **35. Reproducing HotFlip for Corpus Poisoning Attacks in Dense Retrieval**

cs.IR

This paper has been accepted for oral presentation in the  reproducibility track at ECIR 2025

**SubmitDate**: 2025-03-08    [abs](http://arxiv.org/abs/2501.04802v2) [paper-pdf](http://arxiv.org/pdf/2501.04802v2)

**Authors**: Yongkang Li, Panagiotis Eustratiadis, Evangelos Kanoulas

**Abstract**: HotFlip is a topical gradient-based word substitution method for attacking language models. Recently, this method has been further applied to attack retrieval systems by generating malicious passages that are injected into a corpus, i.e., corpus poisoning. However, HotFlip is known to be computationally inefficient, with the majority of time being spent on gradient accumulation for each query-passage pair during the adversarial token generation phase, making it impossible to generate an adequate number of adversarial passages in a reasonable amount of time. Moreover, the attack method itself assumes access to a set of user queries, a strong assumption that does not correspond to how real-world adversarial attacks are usually performed. In this paper, we first significantly boost the efficiency of HotFlip, reducing the adversarial generation process from 4 hours per document to only 15 minutes, using the same hardware. We further contribute experiments and analysis on two additional tasks: (1) transfer-based black-box attacks, and (2) query-agnostic attacks. Whenever possible, we provide comparisons between the original method and our improved version. Our experiments demonstrate that HotFlip can effectively attack a variety of dense retrievers, with an observed trend that its attack performance diminishes against more advanced and recent methods. Interestingly, we observe that while HotFlip performs poorly in a black-box setting, indicating limited capacity for generalization, in query-agnostic scenarios its performance is correlated to the volume of injected adversarial passages.



## **36. IDEATOR: Jailbreaking and Benchmarking Large Vision-Language Models Using Themselves**

cs.CV

**SubmitDate**: 2025-03-08    [abs](http://arxiv.org/abs/2411.00827v3) [paper-pdf](http://arxiv.org/pdf/2411.00827v3)

**Authors**: Ruofan Wang, Juncheng Li, Yixu Wang, Bo Wang, Xiaosen Wang, Yan Teng, Yingchun Wang, Xingjun Ma, Yu-Gang Jiang

**Abstract**: As large Vision-Language Models (VLMs) gain prominence, ensuring their safe deployment has become critical. Recent studies have explored VLM robustness against jailbreak attacks-techniques that exploit model vulnerabilities to elicit harmful outputs. However, the limited availability of diverse multimodal data has constrained current approaches to rely heavily on adversarial or manually crafted images derived from harmful text datasets, which often lack effectiveness and diversity across different contexts. In this paper, we propose IDEATOR, a novel jailbreak method that autonomously generates malicious image-text pairs for black-box jailbreak attacks. IDEATOR is grounded in the insight that VLMs themselves could serve as powerful red team models for generating multimodal jailbreak prompts. Specifically, IDEATOR leverages a VLM to create targeted jailbreak texts and pairs them with jailbreak images generated by a state-of-the-art diffusion model. Extensive experiments demonstrate IDEATOR's high effectiveness and transferability, achieving a 94% attack success rate (ASR) in jailbreaking MiniGPT-4 with an average of only 5.34 queries, and high ASRs of 82%, 88%, and 75% when transferred to LLaVA, InstructBLIP, and Chameleon, respectively. Building on IDEATOR's strong transferability and automated process, we introduce the VLBreakBench, a safety benchmark comprising 3,654 multimodal jailbreak samples. Our benchmark results on 11 recently released VLMs reveal significant gaps in safety alignment. For instance, our challenge set achieves ASRs of 46.31% on GPT-4o and 19.65% on Claude-3.5-Sonnet, underscoring the urgent need for stronger defenses.



## **37. Exploring Adversarial Transferability between Kolmogorov-arnold Networks**

cs.CV

**SubmitDate**: 2025-03-08    [abs](http://arxiv.org/abs/2503.06276v1) [paper-pdf](http://arxiv.org/pdf/2503.06276v1)

**Authors**: Songping Wang, Xinquan Yue, Yueming Lyu, Caifeng Shan

**Abstract**: Kolmogorov-Arnold Networks (KANs) have emerged as a transformative model paradigm, significantly impacting various fields. However, their adversarial robustness remains less underexplored, especially across different KAN architectures. To explore this critical safety issue, we conduct an analysis and find that due to overfitting to the specific basis functions of KANs, they possess poor adversarial transferability among different KANs. To tackle this challenge, we propose AdvKAN, the first transfer attack method for KANs. AdvKAN integrates two key components: 1) a Breakthrough-Defense Surrogate Model (BDSM), which employs a breakthrough-defense training strategy to mitigate overfitting to the specific structures of KANs. 2) a Global-Local Interaction (GLI) technique, which promotes sufficient interaction between adversarial gradients of hierarchical levels, further smoothing out loss surfaces of KANs. Both of them work together to enhance the strength of transfer attack among different KANs. Extensive experimental results on various KANs and datasets demonstrate the effectiveness of AdvKAN, which possesses notably superior attack capabilities and deeply reveals the vulnerabilities of KANs. Code will be released upon acceptance.



## **38. Using Mechanistic Interpretability to Craft Adversarial Attacks against Large Language Models**

cs.LG

**SubmitDate**: 2025-03-08    [abs](http://arxiv.org/abs/2503.06269v1) [paper-pdf](http://arxiv.org/pdf/2503.06269v1)

**Authors**: Thomas Winninger, Boussad Addad, Katarzyna Kapusta

**Abstract**: Traditional white-box methods for creating adversarial perturbations against LLMs typically rely only on gradient computation from the targeted model, ignoring the internal mechanisms responsible for attack success or failure. Conversely, interpretability studies that analyze these internal mechanisms lack practical applications beyond runtime interventions. We bridge this gap by introducing a novel white-box approach that leverages mechanistic interpretability techniques to craft practical adversarial inputs. Specifically, we first identify acceptance subspaces - sets of feature vectors that do not trigger the model's refusal mechanisms - then use gradient-based optimization to reroute embeddings from refusal subspaces to acceptance subspaces, effectively achieving jailbreaks. This targeted approach significantly reduces computation cost, achieving attack success rates of 80-95\% on state-of-the-art models including Gemma2, Llama3.2, and Qwen2.5 within minutes or even seconds, compared to existing techniques that often fail or require hours of computation. We believe this approach opens a new direction for both attack research and defense development. Furthermore, it showcases a practical application of mechanistic interpretability where other methods are less efficient, which highlights its utility. The code and generated datasets are available at https://github.com/Sckathach/subspace-rerouting.



## **39. MUNBa: Machine Unlearning via Nash Bargaining**

cs.CV

**SubmitDate**: 2025-03-08    [abs](http://arxiv.org/abs/2411.15537v2) [paper-pdf](http://arxiv.org/pdf/2411.15537v2)

**Authors**: Jing Wu, Mehrtash Harandi

**Abstract**: Machine Unlearning (MU) aims to selectively erase harmful behaviors from models while retaining the overall utility of the model. As a multi-task learning problem, MU involves balancing objectives related to forgetting specific concepts/data and preserving general performance. A naive integration of these forgetting and preserving objectives can lead to gradient conflicts and dominance, impeding MU algorithms from reaching optimal solutions. To address the gradient conflict and dominance issue, we reformulate MU as a two-player cooperative game, where the two players, namely, the forgetting player and the preservation player, contribute via their gradient proposals to maximize their overall gain and balance their contributions. To this end, inspired by the Nash bargaining theory, we derive a closed-form solution to guide the model toward the Pareto stationary point. Our formulation of MU guarantees an equilibrium solution, where any deviation from the final state would lead to a reduction in the overall objectives for both players, ensuring optimality in each objective. We evaluate our algorithm's effectiveness on a diverse set of tasks across image classification and image generation. Extensive experiments with ResNet, vision-language model CLIP, and text-to-image diffusion models demonstrate that our method outperforms state-of-the-art MU algorithms, achieving a better trade-off between forgetting and preserving. Our results also highlight improvements in forgetting precision, preservation of generalization, and robustness against adversarial attacks.



## **40. Boosting the Local Invariance for Better Adversarial Transferability**

cs.CV

**SubmitDate**: 2025-03-08    [abs](http://arxiv.org/abs/2503.06140v1) [paper-pdf](http://arxiv.org/pdf/2503.06140v1)

**Authors**: Bohan Liu, Xiaosen Wang

**Abstract**: Transfer-based attacks pose a significant threat to real-world applications by directly targeting victim models with adversarial examples generated on surrogate models. While numerous approaches have been proposed to enhance adversarial transferability, existing works often overlook the intrinsic relationship between adversarial perturbations and input images. In this work, we find that adversarial perturbation often exhibits poor translation invariance for a given clean image and model, which is attributed to local invariance. Through empirical analysis, we demonstrate that there is a positive correlation between the local invariance of adversarial perturbations w.r.t. the input image and their transferability across different models. Based on this finding, we propose a general adversarial transferability boosting technique called Local Invariance Boosting approach (LI-Boost). Extensive experiments on the standard ImageNet dataset demonstrate that LI-Boost could significantly boost various types of transfer-based attacks (e.g., gradient-based, input transformation-based, model-related, advanced objective function, ensemble, etc.) on CNNs, ViTs, and defense mechanisms. Our approach presents a promising direction for future research in improving adversarial transferability across different models.



## **41. Continual Adversarial Defense**

cs.CV

**SubmitDate**: 2025-03-07    [abs](http://arxiv.org/abs/2312.09481v5) [paper-pdf](http://arxiv.org/pdf/2312.09481v5)

**Authors**: Qian Wang, Hefei Ling, Yingwei Li, Qihao Liu, Ruoxi Jia, Ning Yu

**Abstract**: In response to the rapidly evolving nature of adversarial attacks against visual classifiers on a monthly basis, numerous defenses have been proposed to generalize against as many known attacks as possible. However, designing a defense method that generalizes to all types of attacks is not realistic because the environment in which defense systems operate is dynamic and comprises various unique attacks that emerge as time goes on. A well-matched approach to the dynamic environment lies in a defense system that continuously collects adversarial data online to quickly improve itself. Therefore, we put forward a practical defense deployment against a challenging threat model and propose, for the first time, the Continual Adversarial Defense (CAD) framework that adapts to attack sequences under four principles: (1)~continual adaptation to new attacks without catastrophic forgetting, (2)~few-shot adaptation, (3)~memory-efficient adaptation, and (4)~high accuracy on both clean and adversarial data. We explore and integrate cutting-edge continual learning, few-shot learning, and ensemble learning techniques to qualify the principles. Extensive experiments validate the effectiveness of our approach against multiple stages of modern adversarial attacks and demonstrate significant improvements over numerous baseline methods. In particular, CAD is capable of quickly adapting with minimal budget and a low cost of defense failure while maintaining good performance against previous attacks. Our research sheds light on a brand-new paradigm for continual defense adaptation against dynamic and evolving attacks.



## **42. Mind the Gap: Detecting Black-box Adversarial Attacks in the Making through Query Update Analysis**

cs.CR

13 pages

**SubmitDate**: 2025-03-07    [abs](http://arxiv.org/abs/2503.02986v2) [paper-pdf](http://arxiv.org/pdf/2503.02986v2)

**Authors**: Jeonghwan Park, Niall McLaughlin, Ihsen Alouani

**Abstract**: Adversarial attacks remain a significant threat that can jeopardize the integrity of Machine Learning (ML) models. In particular, query-based black-box attacks can generate malicious noise without having access to the victim model's architecture, making them practical in real-world contexts. The community has proposed several defenses against adversarial attacks, only to be broken by more advanced and adaptive attack strategies. In this paper, we propose a framework that detects if an adversarial noise instance is being generated. Unlike existing stateful defenses that detect adversarial noise generation by monitoring the input space, our approach learns adversarial patterns in the input update similarity space. In fact, we propose to observe a new metric called Delta Similarity (DS), which we show it captures more efficiently the adversarial behavior. We evaluate our approach against 8 state-of-the-art attacks, including adaptive attacks, where the adversary is aware of the defense and tries to evade detection. We find that our approach is significantly more robust than existing defenses both in terms of specificity and sensitivity.



## **43. Benchmarking Vision Language Model Unlearning via Fictitious Facial Identity Dataset**

cs.CV

**SubmitDate**: 2025-03-07    [abs](http://arxiv.org/abs/2411.03554v3) [paper-pdf](http://arxiv.org/pdf/2411.03554v3)

**Authors**: Yingzi Ma, Jiongxiao Wang, Fei Wang, Siyuan Ma, Jiazhao Li, Jinsheng Pan, Xiujun Li, Furong Huang, Lichao Sun, Bo Li, Yejin Choi, Muhao Chen, Chaowei Xiao

**Abstract**: Machine unlearning has emerged as an effective strategy for forgetting specific information in the training data. However, with the increasing integration of visual data, privacy concerns in Vision Language Models (VLMs) remain underexplored. To address this, we introduce Facial Identity Unlearning Benchmark (FIUBench), a novel VLM unlearning benchmark designed to robustly evaluate the effectiveness of unlearning algorithms under the Right to be Forgotten setting. Specifically, we formulate the VLM unlearning task via constructing the Fictitious Facial Identity VQA dataset and apply a two-stage evaluation pipeline that is designed to precisely control the sources of information and their exposure levels. In terms of evaluation, since VLM supports various forms of ways to ask questions with the same semantic meaning, we also provide robust evaluation metrics including membership inference attacks and carefully designed adversarial privacy attacks to evaluate the performance of algorithms. Through the evaluation of four baseline VLM unlearning algorithms within FIUBench, we find that all methods remain limited in their unlearning performance, with significant trade-offs between model utility and forget quality. Furthermore, our findings also highlight the importance of privacy attacks for robust evaluations. We hope FIUBench will drive progress in developing more effective VLM unlearning algorithms.



## **44. Toward Robust Non-Transferable Learning: A Survey and Benchmark**

cs.LG

Code is available at https://github.com/tmllab/NTLBench

**SubmitDate**: 2025-03-07    [abs](http://arxiv.org/abs/2502.13593v2) [paper-pdf](http://arxiv.org/pdf/2502.13593v2)

**Authors**: Ziming Hong, Yongli Xiang, Tongliang Liu

**Abstract**: Over the past decades, researchers have primarily focused on improving the generalization abilities of models, with limited attention given to regulating such generalization. However, the ability of models to generalize to unintended data (e.g., harmful or unauthorized data) can be exploited by malicious adversaries in unforeseen ways, potentially resulting in violations of model ethics. Non-transferable learning (NTL), a task aimed at reshaping the generalization abilities of deep learning models, was proposed to address these challenges. While numerous methods have been proposed in this field, a comprehensive review of existing progress and a thorough analysis of current limitations remain lacking. In this paper, we bridge this gap by presenting the first comprehensive survey on NTL and introducing NTLBench, the first benchmark to evaluate NTL performance and robustness within a unified framework. Specifically, we first introduce the task settings, general framework, and criteria of NTL, followed by a summary of NTL approaches. Furthermore, we emphasize the often-overlooked issue of robustness against various attacks that can destroy the non-transferable mechanism established by NTL. Experiments conducted via NTLBench verify the limitations of existing NTL methods in robustness. Finally, we discuss the practical applications of NTL, along with its future directions and associated challenges.



## **45. Robust Intrusion Detection System with Explainable Artificial Intelligence**

cs.CR

**SubmitDate**: 2025-03-07    [abs](http://arxiv.org/abs/2503.05303v1) [paper-pdf](http://arxiv.org/pdf/2503.05303v1)

**Authors**: Betül Güvenç Paltun, Ramin Fuladi, Rim El Malki

**Abstract**: Machine learning (ML) models serve as powerful tools for threat detection and mitigation; however, they also introduce potential new risks. Adversarial input can exploit these models through standard interfaces, thus creating new attack pathways that threaten critical network operations. As ML advancements progress, adversarial strategies become more advanced, and conventional defenses such as adversarial training are costly in computational terms and often fail to provide real-time detection. These methods typically require a balance between robustness and model performance, which presents challenges for applications that demand instant response. To further investigate this vulnerability, we suggest a novel strategy for detecting and mitigating adversarial attacks using eXplainable Artificial Intelligence (XAI). This approach is evaluated in real time within intrusion detection systems (IDS), leading to the development of a zero-touch mitigation strategy. Additionally, we explore various scenarios in the Radio Resource Control (RRC) layer within the Open Radio Access Network (O-RAN) framework, emphasizing the critical need for enhanced mitigation techniques to strengthen IDS defenses against advanced threats and implement a zero-touch mitigation solution. Extensive testing across different scenarios in the RRC layer of the O-RAN infrastructure validates the ability of the framework to detect and counteract integrated RRC-layer attacks when paired with adversarial strategies, emphasizing the essential need for robust defensive mechanisms to strengthen IDS against complex threats.



## **46. Jailbreaking is (Mostly) Simpler Than You Think**

cs.CR

**SubmitDate**: 2025-03-07    [abs](http://arxiv.org/abs/2503.05264v1) [paper-pdf](http://arxiv.org/pdf/2503.05264v1)

**Authors**: Mark Russinovich, Ahmed Salem

**Abstract**: We introduce the Context Compliance Attack (CCA), a novel, optimization-free method for bypassing AI safety mechanisms. Unlike current approaches -- which rely on complex prompt engineering and computationally intensive optimization -- CCA exploits a fundamental architectural vulnerability inherent in many deployed AI systems. By subtly manipulating conversation history, CCA convinces the model to comply with a fabricated dialogue context, thereby triggering restricted behavior. Our evaluation across a diverse set of open-source and proprietary models demonstrates that this simple attack can circumvent state-of-the-art safety protocols. We discuss the implications of these findings and propose practical mitigation strategies to fortify AI systems against such elementary yet effective adversarial tactics.



## **47. DetectRL: Benchmarking LLM-Generated Text Detection in Real-World Scenarios**

cs.CL

Accepted to NeurIPS 2024 Datasets and Benchmarks Track (Camera-Ready)

**SubmitDate**: 2025-03-07    [abs](http://arxiv.org/abs/2410.23746v2) [paper-pdf](http://arxiv.org/pdf/2410.23746v2)

**Authors**: Junchao Wu, Runzhe Zhan, Derek F. Wong, Shu Yang, Xinyi Yang, Yulin Yuan, Lidia S. Chao

**Abstract**: Detecting text generated by large language models (LLMs) is of great recent interest. With zero-shot methods like DetectGPT, detection capabilities have reached impressive levels. However, the reliability of existing detectors in real-world applications remains underexplored. In this study, we present a new benchmark, DetectRL, highlighting that even state-of-the-art (SOTA) detection techniques still underperformed in this task. We collected human-written datasets from domains where LLMs are particularly prone to misuse. Using popular LLMs, we generated data that better aligns with real-world applications. Unlike previous studies, we employed heuristic rules to create adversarial LLM-generated text, simulating various prompts usages, human revisions like word substitutions, and writing noises like spelling mistakes. Our development of DetectRL reveals the strengths and limitations of current SOTA detectors. More importantly, we analyzed the potential impact of writing styles, model types, attack methods, the text lengths, and real-world human writing factors on different types of detectors. We believe DetectRL could serve as an effective benchmark for assessing detectors in real-world scenarios, evolving with advanced attack methods, thus providing more stressful evaluation to drive the development of more efficient detectors. Data and code are publicly available at: https://github.com/NLP2CT/DetectRL.



## **48. Safety-Critical Traffic Simulation with Adversarial Transfer of Driving Intentions**

cs.RO

Accepted by ICRA 2025

**SubmitDate**: 2025-03-07    [abs](http://arxiv.org/abs/2503.05180v1) [paper-pdf](http://arxiv.org/pdf/2503.05180v1)

**Authors**: Zherui Huang, Xing Gao, Guanjie Zheng, Licheng Wen, Xuemeng Yang, Xiao Sun

**Abstract**: Traffic simulation, complementing real-world data with a long-tail distribution, allows for effective evaluation and enhancement of the ability of autonomous vehicles to handle accident-prone scenarios. Simulating such safety-critical scenarios is nontrivial, however, from log data that are typically regular scenarios, especially in consideration of dynamic adversarial interactions between the future motions of autonomous vehicles and surrounding traffic participants. To address it, this paper proposes an innovative and efficient strategy, termed IntSim, that explicitly decouples the driving intentions of surrounding actors from their motion planning for realistic and efficient safety-critical simulation. We formulate the adversarial transfer of driving intention as an optimization problem, facilitating extensive exploration of diverse attack behaviors and efficient solution convergence. Simultaneously, intention-conditioned motion planning benefits from powerful deep models and large-scale real-world data, permitting the simulation of realistic motion behaviors for actors. Specially, through adapting driving intentions based on environments, IntSim facilitates the flexible realization of dynamic adversarial interactions with autonomous vehicles. Finally, extensive open-loop and closed-loop experiments on real-world datasets, including nuScenes and Waymo, demonstrate that the proposed IntSim achieves state-of-the-art performance in simulating realistic safety-critical scenarios and further improves planners in handling such scenarios.



## **49. Double Backdoored: Converting Code Large Language Model Backdoors to Traditional Malware via Adversarial Instruction Tuning Attacks**

cs.CR

**SubmitDate**: 2025-03-07    [abs](http://arxiv.org/abs/2404.18567v2) [paper-pdf](http://arxiv.org/pdf/2404.18567v2)

**Authors**: Md Imran Hossen, Sai Venkatesh Chilukoti, Liqun Shan, Sheng Chen, Yinzhi Cao, Xiali Hei

**Abstract**: Instruction-tuned Large Language Models designed for coding tasks are increasingly employed as AI coding assistants. However, the cybersecurity vulnerabilities and implications arising from the widespread integration of these models are not yet fully understood due to limited research in this domain. This work investigates novel techniques for transitioning backdoors from the AI/ML domain to traditional computer malware, shedding light on the critical intersection of AI and cyber/software security. To explore this intersection, we present MalInstructCoder, a framework designed to comprehensively assess the cybersecurity vulnerabilities of instruction-tuned Code LLMs. MalInstructCoder introduces an automated data poisoning pipeline to inject malicious code snippets into benign code, poisoning instruction fine-tuning data while maintaining functional validity. It presents two practical adversarial instruction tuning attacks with real-world security implications: the clean prompt poisoning attack and the backdoor attack. These attacks aim to manipulate Code LLMs to generate code incorporating malicious or harmful functionality under specific attack scenarios while preserving intended functionality. We conduct a comprehensive investigation into the exploitability of the code-specific instruction tuning process involving three state-of-the-art Code LLMs: CodeLlama, DeepSeek-Coder, and StarCoder2. Our findings reveal that these models are highly vulnerable to our attacks. Specifically, the clean prompt poisoning attack achieves the ASR@1 ranging from over 75% to 86% by poisoning only 1% (162 samples) of the instruction fine-tuning dataset. Similarly, the backdoor attack achieves the ASR@1 ranging from 76% to 86% with a 0.5% poisoning rate. Our study sheds light on the critical cybersecurity risks posed by instruction-tuned Code LLMs and highlights the urgent need for robust defense mechanisms.



## **50. Safety is Not Only About Refusal: Reasoning-Enhanced Fine-tuning for Interpretable LLM Safety**

cs.CL

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2503.05021v1) [paper-pdf](http://arxiv.org/pdf/2503.05021v1)

**Authors**: Yuyou Zhang, Miao Li, William Han, Yihang Yao, Zhepeng Cen, Ding Zhao

**Abstract**: Large Language Models (LLMs) are vulnerable to jailbreak attacks that exploit weaknesses in traditional safety alignment, which often relies on rigid refusal heuristics or representation engineering to block harmful outputs. While they are effective for direct adversarial attacks, they fall short of broader safety challenges requiring nuanced, context-aware decision-making. To address this, we propose Reasoning-enhanced Finetuning for interpretable LLM Safety (Rational), a novel framework that trains models to engage in explicit safe reasoning before response. Fine-tuned models leverage the extensive pretraining knowledge in self-generated reasoning to bootstrap their own safety through structured reasoning, internalizing context-sensitive decision-making. Our findings suggest that safety extends beyond refusal, requiring context awareness for more robust, interpretable, and adaptive responses. Reasoning is not only a core capability of LLMs but also a fundamental mechanism for LLM safety. Rational employs reasoning-enhanced fine-tuning, allowing it to reject harmful prompts while providing meaningful and context-aware responses in complex scenarios.



