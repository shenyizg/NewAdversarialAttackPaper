# Latest Large Language Model Attack Papers
**update at 2025-03-21 10:39:35**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. Immune: Improving Safety Against Jailbreaks in Multi-modal LLMs via Inference-Time Alignment**

cs.CR

Accepted to CVPR 2025

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2411.18688v3) [paper-pdf](http://arxiv.org/pdf/2411.18688v3)

**Authors**: Soumya Suvra Ghosal, Souradip Chakraborty, Vaibhav Singh, Tianrui Guan, Mengdi Wang, Ahmad Beirami, Furong Huang, Alvaro Velasquez, Dinesh Manocha, Amrit Singh Bedi

**Abstract**: With the widespread deployment of Multimodal Large Language Models (MLLMs) for visual-reasoning tasks, improving their safety has become crucial. Recent research indicates that despite training-time safety alignment, these models remain vulnerable to jailbreak attacks. In this work, we first highlight an important safety gap to describe that alignment achieved solely through safety training may be insufficient against jailbreak attacks. To address this vulnerability, we propose Immune, an inference-time defense framework that leverages a safe reward model through controlled decoding to defend against jailbreak attacks. Additionally, we provide a mathematical characterization of Immune, offering insights on why it improves safety against jailbreaks. Extensive evaluations on diverse jailbreak benchmarks using recent MLLMs reveal that Immune effectively enhances model safety while preserving the model's original capabilities. For instance, against text-based jailbreak attacks on LLaVA-1.6, Immune reduces the attack success rate by 57.82% and 16.78% compared to the base MLLM and state-of-the-art defense strategy, respectively.



## **2. Robust LLM safeguarding via refusal feature adversarial training**

cs.LG

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2409.20089v2) [paper-pdf](http://arxiv.org/pdf/2409.20089v2)

**Authors**: Lei Yu, Virginie Do, Karen Hambardzumyan, Nicola Cancedda

**Abstract**: Large language models (LLMs) are vulnerable to adversarial attacks that can elicit harmful responses. Defending against such attacks remains challenging due to the opacity of jailbreaking mechanisms and the high computational cost of training LLMs robustly. We demonstrate that adversarial attacks share a universal mechanism for circumventing LLM safeguards that works by ablating a dimension in the residual stream embedding space called the refusal feature. We further show that the operation of refusal feature ablation (RFA) approximates the worst-case perturbation of offsetting model safety. Based on these findings, we propose Refusal Feature Adversarial Training (ReFAT), a novel algorithm that efficiently performs LLM adversarial training by simulating the effect of input-level attacks via RFA. Experiment results show that ReFAT significantly improves the robustness of three popular LLMs against a wide range of adversarial attacks, with considerably less computational overhead compared to existing adversarial training methods.



## **3. "Moralized" Multi-Step Jailbreak Prompts: Black-Box Testing of Guardrails in Large Language Models for Verbal Attacks**

cs.CR

This paper has been submitted to Nature Machine Intelligence and  OpenReview preprints. It has 7 pages of text, 3 figures, and 3 tables

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2411.16730v4) [paper-pdf](http://arxiv.org/pdf/2411.16730v4)

**Authors**: Libo Wang

**Abstract**: As the application of large language models continues to expand in various fields, it poses higher challenges to the effectiveness of identifying harmful content generation and guardrail mechanisms. This research aims to evaluate the guardrail effectiveness of GPT-4o, Grok-2 Beta, Llama 3.1 (405B), Gemini 1.5, and Claude 3.5 Sonnet through black-box testing of seemingly ethical multi-step jailbreak prompts. It conducts ethical attacks by designing an identical multi-step prompts that simulates the scenario of "corporate middle managers competing for promotions." The data results show that the guardrails of the above-mentioned LLMs were bypassed and the content of verbal attacks was generated. Claude 3.5 Sonnet's resistance to multi-step jailbreak prompts is more obvious. To ensure objectivity, the experimental process, black box test code, and enhanced guardrail code are uploaded to the GitHub repository: https://github.com/brucewang123456789/GeniusTrail.git.



## **4. BadToken: Token-level Backdoor Attacks to Multi-modal Large Language Models**

cs.CR

This paper is accepted by CVPR 2025

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2503.16023v1) [paper-pdf](http://arxiv.org/pdf/2503.16023v1)

**Authors**: Zenghui Yuan, Jiawen Shi, Pan Zhou, Neil Zhenqiang Gong, Lichao Sun

**Abstract**: Multi-modal large language models (MLLMs) extend large language models (LLMs) to process multi-modal information, enabling them to generate responses to image-text inputs. MLLMs have been incorporated into diverse multi-modal applications, such as autonomous driving and medical diagnosis, via plug-and-play without fine-tuning. This deployment paradigm increases the vulnerability of MLLMs to backdoor attacks. However, existing backdoor attacks against MLLMs achieve limited effectiveness and stealthiness. In this work, we propose BadToken, the first token-level backdoor attack to MLLMs. BadToken introduces two novel backdoor behaviors: Token-substitution and Token-addition, which enable flexible and stealthy attacks by making token-level modifications to the original output for backdoored inputs. We formulate a general optimization problem that considers the two backdoor behaviors to maximize the attack effectiveness. We evaluate BadToken on two open-source MLLMs and various tasks. Our results show that our attack maintains the model's utility while achieving high attack success rates and stealthiness. We also show the real-world threats of BadToken in two scenarios, i.e., autonomous driving and medical diagnosis. Furthermore, we consider defenses including fine-tuning and input purification. Our results highlight the threat of our attack.



## **5. Differentially Private Steering for Large Language Model Alignment**

cs.CL

ICLR 2025 Camera Ready; Code: https://github.com/UKPLab/iclr2025-psa

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2501.18532v2) [paper-pdf](http://arxiv.org/pdf/2501.18532v2)

**Authors**: Anmol Goel, Yaxi Hu, Iryna Gurevych, Amartya Sanyal

**Abstract**: Aligning Large Language Models (LLMs) with human values and away from undesirable behaviors (such as hallucination) has become increasingly important. Recently, steering LLMs towards a desired behavior via activation editing has emerged as an effective method to mitigate harmful generations at inference-time. Activation editing modifies LLM representations by preserving information from positive demonstrations (e.g., truthful) and minimising information from negative demonstrations (e.g., hallucinations). When these demonstrations come from a private dataset, the aligned LLM may leak private information contained in those private samples. In this work, we present the first study of aligning LLM behavior with private datasets. Our work proposes the Private Steering for LLM Alignment (PSA) algorithm to edit LLM activations with differential privacy (DP) guarantees. We conduct extensive experiments on seven different benchmarks with open-source LLMs of different sizes (0.5B to 7B) and model families (LlaMa, Qwen, Mistral and Gemma). Our results show that PSA achieves DP guarantees for LLM alignment with minimal loss in performance, including alignment metrics, open-ended text generation quality, and general-purpose reasoning. We also develop the first Membership Inference Attack (MIA) for evaluating and auditing the empirical privacy for the problem of LLM steering via activation editing. Our experiments support the theoretical guarantees by showing improved guarantees for our PSA algorithm compared to several existing non-private techniques.



## **6. SAUCE: Selective Concept Unlearning in Vision-Language Models with Sparse Autoencoders**

cs.CV

More comparative experiments are needed

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2503.14530v2) [paper-pdf](http://arxiv.org/pdf/2503.14530v2)

**Authors**: Qing Li, Jiahui Geng, Derui Zhu, Fengyu Cai, Chenyang Lyu, Fakhri Karray

**Abstract**: Unlearning methods for vision-language models (VLMs) have primarily adapted techniques from large language models (LLMs), relying on weight updates that demand extensive annotated forget sets. Moreover, these methods perform unlearning at a coarse granularity, often leading to excessive forgetting and reduced model utility. To address this issue, we introduce SAUCE, a novel method that leverages sparse autoencoders (SAEs) for fine-grained and selective concept unlearning in VLMs. Briefly, SAUCE first trains SAEs to capture high-dimensional, semantically rich sparse features. It then identifies the features most relevant to the target concept for unlearning. During inference, it selectively modifies these features to suppress specific concepts while preserving unrelated information. We evaluate SAUCE on two distinct VLMs, LLaVA-v1.5-7B and LLaMA-3.2-11B-Vision-Instruct, across two types of tasks: concrete concept unlearning (objects and sports scenes) and abstract concept unlearning (emotions, colors, and materials), encompassing a total of 60 concepts. Extensive experiments demonstrate that SAUCE outperforms state-of-the-art methods by 18.04% in unlearning quality while maintaining comparable model utility. Furthermore, we investigate SAUCE's robustness against widely used adversarial attacks, its transferability across models, and its scalability in handling multiple simultaneous unlearning requests. Our findings establish SAUCE as an effective and scalable solution for selective concept unlearning in VLMs.



## **7. DroidTTP: Mapping Android Applications with TTP for Cyber Threat Intelligence**

cs.CR

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2503.15866v1) [paper-pdf](http://arxiv.org/pdf/2503.15866v1)

**Authors**: Dincy R Arikkat, Vinod P., Rafidha Rehiman K. A., Serena Nicolazzo, Marco Arazzi, Antonino Nocera, Mauro Conti

**Abstract**: The widespread adoption of Android devices for sensitive operations like banking and communication has made them prime targets for cyber threats, particularly Advanced Persistent Threats (APT) and sophisticated malware attacks. Traditional malware detection methods rely on binary classification, failing to provide insights into adversarial Tactics, Techniques, and Procedures (TTPs). Understanding malware behavior is crucial for enhancing cybersecurity defenses. To address this gap, we introduce DroidTTP, a framework mapping Android malware behaviors to TTPs based on the MITRE ATT&CK framework. Our curated dataset explicitly links MITRE TTPs to Android applications. We developed an automated solution leveraging the Problem Transformation Approach (PTA) and Large Language Models (LLMs) to map applications to both Tactics and Techniques. Additionally, we employed Retrieval-Augmented Generation (RAG) with prompt engineering and LLM fine-tuning for TTP predictions. Our structured pipeline includes dataset creation, hyperparameter tuning, data augmentation, feature selection, model development, and SHAP-based model interpretability. Among LLMs, Llama achieved the highest performance in Tactic classification with a Jaccard Similarity of 0.9583 and Hamming Loss of 0.0182, and in Technique classification with a Jaccard Similarity of 0.9348 and Hamming Loss of 0.0127. However, the Label Powerset XGBoost model outperformed LLMs, achieving a Jaccard Similarity of 0.9893 for Tactic classification and 0.9753 for Technique classification, with a Hamming Loss of 0.0054 and 0.0050, respectively. While XGBoost showed superior performance, the narrow margin highlights the potential of LLM-based approaches in TTP classification.



## **8. AutoRedTeamer: Autonomous Red Teaming with Lifelong Attack Integration**

cs.CR

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2503.15754v1) [paper-pdf](http://arxiv.org/pdf/2503.15754v1)

**Authors**: Andy Zhou, Kevin Wu, Francesco Pinto, Zhaorun Chen, Yi Zeng, Yu Yang, Shuang Yang, Sanmi Koyejo, James Zou, Bo Li

**Abstract**: As large language models (LLMs) become increasingly capable, security and safety evaluation are crucial. While current red teaming approaches have made strides in assessing LLM vulnerabilities, they often rely heavily on human input and lack comprehensive coverage of emerging attack vectors. This paper introduces AutoRedTeamer, a novel framework for fully automated, end-to-end red teaming against LLMs. AutoRedTeamer combines a multi-agent architecture with a memory-guided attack selection mechanism to enable continuous discovery and integration of new attack vectors. The dual-agent framework consists of a red teaming agent that can operate from high-level risk categories alone to generate and execute test cases and a strategy proposer agent that autonomously discovers and implements new attacks by analyzing recent research. This modular design allows AutoRedTeamer to adapt to emerging threats while maintaining strong performance on existing attack vectors. We demonstrate AutoRedTeamer's effectiveness across diverse evaluation settings, achieving 20% higher attack success rates on HarmBench against Llama-3.1-70B while reducing computational costs by 46% compared to existing approaches. AutoRedTeamer also matches the diversity of human-curated benchmarks in generating test cases, providing a comprehensive, scalable, and continuously evolving framework for evaluating the security of AI systems.



## **9. Undesirable Memorization in Large Language Models: A Survey**

cs.CL

**SubmitDate**: 2025-03-19    [abs](http://arxiv.org/abs/2410.02650v2) [paper-pdf](http://arxiv.org/pdf/2410.02650v2)

**Authors**: Ali Satvaty, Suzan Verberne, Fatih Turkmen

**Abstract**: While recent research increasingly showcases the remarkable capabilities of Large Language Models (LLMs), it is equally crucial to examine their associated risks. Among these, privacy and security vulnerabilities are particularly concerning, posing significant ethical and legal challenges. At the heart of these vulnerabilities stands memorization, which refers to a model's tendency to store and reproduce phrases from its training data. This phenomenon has been shown to be a fundamental source to various privacy and security attacks against LLMs. In this paper, we provide a taxonomy of the literature on LLM memorization, exploring it across three dimensions: granularity, retrievability, and desirability. Next, we discuss the metrics and methods used to quantify memorization, followed by an analysis of the causes and factors that contribute to memorization phenomenon. We then explore strategies that are used so far to mitigate the undesirable aspects of this phenomenon. We conclude our survey by identifying potential research topics for the near future, including methods to balance privacy and performance, and the analysis of memorization in specific LLM contexts such as conversational agents, retrieval-augmented generation, and diffusion language models. Given the rapid research pace in this field, we also maintain a dedicated repository of the references discussed in this survey which will be regularly updated to reflect the latest developments.



## **10. Safety at Scale: A Comprehensive Survey of Large Model Safety**

cs.CR

47 pages, 3 figures, 11 tables; GitHub:  https://github.com/xingjunm/Awesome-Large-Model-Safety

**SubmitDate**: 2025-03-19    [abs](http://arxiv.org/abs/2502.05206v3) [paper-pdf](http://arxiv.org/pdf/2502.05206v3)

**Authors**: Xingjun Ma, Yifeng Gao, Yixu Wang, Ruofan Wang, Xin Wang, Ye Sun, Yifan Ding, Hengyuan Xu, Yunhao Chen, Yunhan Zhao, Hanxun Huang, Yige Li, Jiaming Zhang, Xiang Zheng, Yang Bai, Zuxuan Wu, Xipeng Qiu, Jingfeng Zhang, Yiming Li, Xudong Han, Haonan Li, Jun Sun, Cong Wang, Jindong Gu, Baoyuan Wu, Siheng Chen, Tianwei Zhang, Yang Liu, Mingming Gong, Tongliang Liu, Shirui Pan, Cihang Xie, Tianyu Pang, Yinpeng Dong, Ruoxi Jia, Yang Zhang, Shiqing Ma, Xiangyu Zhang, Neil Gong, Chaowei Xiao, Sarah Erfani, Tim Baldwin, Bo Li, Masashi Sugiyama, Dacheng Tao, James Bailey, Yu-Gang Jiang

**Abstract**: The rapid advancement of large models, driven by their exceptional abilities in learning and generalization through large-scale pre-training, has reshaped the landscape of Artificial Intelligence (AI). These models are now foundational to a wide range of applications, including conversational AI, recommendation systems, autonomous driving, content generation, medical diagnostics, and scientific discovery. However, their widespread deployment also exposes them to significant safety risks, raising concerns about robustness, reliability, and ethical implications. This survey provides a systematic review of current safety research on large models, covering Vision Foundation Models (VFMs), Large Language Models (LLMs), Vision-Language Pre-training (VLP) models, Vision-Language Models (VLMs), Diffusion Models (DMs), and large-model-based Agents. Our contributions are summarized as follows: (1) We present a comprehensive taxonomy of safety threats to these models, including adversarial attacks, data poisoning, backdoor attacks, jailbreak and prompt injection attacks, energy-latency attacks, data and model extraction attacks, and emerging agent-specific threats. (2) We review defense strategies proposed for each type of attacks if available and summarize the commonly used datasets and benchmarks for safety research. (3) Building on this, we identify and discuss the open challenges in large model safety, emphasizing the need for comprehensive safety evaluations, scalable and effective defense mechanisms, and sustainable data practices. More importantly, we highlight the necessity of collective efforts from the research community and international collaboration. Our work can serve as a useful reference for researchers and practitioners, fostering the ongoing development of comprehensive defense systems and platforms to safeguard AI models.



## **11. Assessing AI vs Human-Authored Spear Phishing SMS Attacks: An Empirical Study**

cs.CY

18 pages, 5 figures, 1 table

**SubmitDate**: 2025-03-19    [abs](http://arxiv.org/abs/2406.13049v2) [paper-pdf](http://arxiv.org/pdf/2406.13049v2)

**Authors**: Jerson Francia, Derek Hansen, Ben Schooley, Matthew Taylor, Shydra Murray, Greg Snow

**Abstract**: This paper explores the use of Large Language Models (LLMs) in spear phishing message generation and evaluates their performance compared to human-authored counterparts. Our pilot study examines the effectiveness of smishing (SMS phishing) messages created by GPT-4 and human authors, which have been personalized for willing targets. The targets assessed these messages in a modified ranked-order experiment using a novel methodology we call TRAPD (Threshold Ranking Approach for Personalized Deception). Experiments involved ranking each spear phishing message from most to least convincing, providing qualitative feedback, and guessing which messages were human- or AI-generated. Results show that LLM-generated messages are often perceived as more convincing than those authored by humans, particularly job-related messages. Targets also struggled to distinguish between human- and AI-generated messages. We analyze different criteria the targets used to assess the persuasiveness and source of messages. This study aims to highlight the urgent need for further research and improved countermeasures against personalized AI-enabled social engineering attacks.



## **12. Temporal Context Awareness: A Defense Framework Against Multi-turn Manipulation Attacks on Large Language Models**

cs.CR

6 pages, 2 figures, IEEE CAI

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2503.15560v1) [paper-pdf](http://arxiv.org/pdf/2503.15560v1)

**Authors**: Prashant Kulkarni, Assaf Namer

**Abstract**: Large Language Models (LLMs) are increasingly vulnerable to sophisticated multi-turn manipulation attacks, where adversaries strategically build context through seemingly benign conversational turns to circumvent safety measures and elicit harmful or unauthorized responses. These attacks exploit the temporal nature of dialogue to evade single-turn detection methods, representing a critical security vulnerability with significant implications for real-world deployments.   This paper introduces the Temporal Context Awareness (TCA) framework, a novel defense mechanism designed to address this challenge by continuously analyzing semantic drift, cross-turn intention consistency and evolving conversational patterns. The TCA framework integrates dynamic context embedding analysis, cross-turn consistency verification, and progressive risk scoring to detect and mitigate manipulation attempts effectively. Preliminary evaluations on simulated adversarial scenarios demonstrate the framework's potential to identify subtle manipulation patterns often missed by traditional detection techniques, offering a much-needed layer of security for conversational AI systems. In addition to outlining the design of TCA , we analyze diverse attack vectors and their progression across multi-turn conversation, providing valuable insights into adversarial tactics and their impact on LLM vulnerabilities. Our findings underscore the pressing need for robust, context-aware defenses in conversational AI systems and highlight TCA framework as a promising direction for securing LLMs while preserving their utility in legitimate applications. We make our implementation available to support further research in this emerging area of AI security.



## **13. Personalized Attacks of Social Engineering in Multi-turn Conversations -- LLM Agents for Simulation and Detection**

cs.CR

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2503.15552v1) [paper-pdf](http://arxiv.org/pdf/2503.15552v1)

**Authors**: Tharindu Kumarage, Cameron Johnson, Jadie Adams, Lin Ai, Matthias Kirchner, Anthony Hoogs, Joshua Garland, Julia Hirschberg, Arslan Basharat, Huan Liu

**Abstract**: The rapid advancement of conversational agents, particularly chatbots powered by Large Language Models (LLMs), poses a significant risk of social engineering (SE) attacks on social media platforms. SE detection in multi-turn, chat-based interactions is considerably more complex than single-instance detection due to the dynamic nature of these conversations. A critical factor in mitigating this threat is understanding the mechanisms through which SE attacks operate, specifically how attackers exploit vulnerabilities and how victims' personality traits contribute to their susceptibility. In this work, we propose an LLM-agentic framework, SE-VSim, to simulate SE attack mechanisms by generating multi-turn conversations. We model victim agents with varying personality traits to assess how psychological profiles influence susceptibility to manipulation. Using a dataset of over 1000 simulated conversations, we examine attack scenarios in which adversaries, posing as recruiters, funding agencies, and journalists, attempt to extract sensitive information. Based on this analysis, we present a proof of concept, SE-OmniGuard, to offer personalized protection to users by leveraging prior knowledge of the victims personality, evaluating attack strategies, and monitoring information exchanges in conversations to identify potential SE attempts.



## **14. TAROT: Task-Oriented Authorship Obfuscation Using Policy Optimization Methods**

cs.CL

Accepted to the NAACL PrivateNLP 2025 Workshop

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2407.21630v2) [paper-pdf](http://arxiv.org/pdf/2407.21630v2)

**Authors**: Gabriel Loiseau, Damien Sileo, Damien Riquet, Maxime Meyer, Marc Tommasi

**Abstract**: Authorship obfuscation aims to disguise the identity of an author within a text by altering the writing style, vocabulary, syntax, and other linguistic features associated with the text author. This alteration needs to balance privacy and utility. While strong obfuscation techniques can effectively hide the author's identity, they often degrade the quality and usefulness of the text for its intended purpose. Conversely, maintaining high utility tends to provide insufficient privacy, making it easier for an adversary to de-anonymize the author. Thus, achieving an optimal trade-off between these two conflicting objectives is crucial. In this paper, we propose TAROT: Task-Oriented Authorship Obfuscation Using Policy Optimization, a new unsupervised authorship obfuscation method whose goal is to optimize the privacy-utility trade-off by regenerating the entire text considering its downstream utility. Our approach leverages policy optimization as a fine-tuning paradigm over small language models in order to rewrite texts by preserving author identity and downstream task utility. We show that our approach largely reduces the accuracy of attackers while preserving utility. We make our code and models publicly available.



## **15. Adversarial Training for Multimodal Large Language Models against Jailbreak Attacks**

cs.CV

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2503.04833v2) [paper-pdf](http://arxiv.org/pdf/2503.04833v2)

**Authors**: Liming Lu, Shuchao Pang, Siyuan Liang, Haotian Zhu, Xiyu Zeng, Aishan Liu, Yunhuai Liu, Yongbin Zhou

**Abstract**: Multimodal large language models (MLLMs) have made remarkable strides in cross-modal comprehension and generation tasks. However, they remain vulnerable to jailbreak attacks, where crafted perturbations bypass security guardrails and elicit harmful outputs. In this paper, we present the first adversarial training (AT) paradigm tailored to defend against jailbreak attacks during the MLLM training phase. Extending traditional AT to this domain poses two critical challenges: efficiently tuning massive parameters and ensuring robustness against attacks across multiple modalities. To address these challenges, we introduce Projection Layer Against Adversarial Training (ProEAT), an end-to-end AT framework. ProEAT incorporates a projector-based adversarial training architecture that efficiently handles large-scale parameters while maintaining computational feasibility by focusing adversarial training on a lightweight projector layer instead of the entire model; additionally, we design a dynamic weight adjustment mechanism that optimizes the loss function's weight allocation based on task demands, streamlining the tuning process. To enhance defense performance, we propose a joint optimization strategy across visual and textual modalities, ensuring robust resistance to jailbreak attacks originating from either modality. Extensive experiments conducted on five major jailbreak attack methods across three mainstream MLLMs demonstrate the effectiveness of our approach. ProEAT achieves state-of-the-art defense performance, outperforming existing baselines by an average margin of +34% across text and image modalities, while incurring only a 1% reduction in clean accuracy. Furthermore, evaluations on real-world embodied intelligent systems highlight the practical applicability of our framework, paving the way for the development of more secure and reliable multimodal systems.



## **16. Survey of Adversarial Robustness in Multimodal Large Language Models**

cs.CV

9 pages

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2503.13962v1) [paper-pdf](http://arxiv.org/pdf/2503.13962v1)

**Authors**: Chengze Jiang, Zhuangzhuang Wang, Minjing Dong, Jie Gui

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated exceptional performance in artificial intelligence by facilitating integrated understanding across diverse modalities, including text, images, video, audio, and speech. However, their deployment in real-world applications raises significant concerns about adversarial vulnerabilities that could compromise their safety and reliability. Unlike unimodal models, MLLMs face unique challenges due to the interdependencies among modalities, making them susceptible to modality-specific threats and cross-modal adversarial manipulations. This paper reviews the adversarial robustness of MLLMs, covering different modalities. We begin with an overview of MLLMs and a taxonomy of adversarial attacks tailored to each modality. Next, we review key datasets and evaluation metrics used to assess the robustness of MLLMs. After that, we provide an in-depth review of attacks targeting MLLMs across different modalities. Our survey also identifies critical challenges and suggests promising future research directions.



## **17. AttackEval: How to Evaluate the Effectiveness of Jailbreak Attacking on Large Language Models**

cs.CL

Accepted by ACM SIGKDD Explorations 2025

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2401.09002v6) [paper-pdf](http://arxiv.org/pdf/2401.09002v6)

**Authors**: Dong Shu, Chong Zhang, Mingyu Jin, Zihao Zhou, Lingyao Li, Yongfeng Zhang

**Abstract**: Jailbreak attacks represent one of the most sophisticated threats to the security of large language models (LLMs). To deal with such risks, we introduce an innovative framework that can help evaluate the effectiveness of jailbreak attacks on LLMs. Unlike traditional binary evaluations focusing solely on the robustness of LLMs, our method assesses the attacking prompts' effectiveness. We present two distinct evaluation frameworks: a coarse-grained evaluation and a fine-grained evaluation. Each framework uses a scoring range from 0 to 1, offering unique perspectives and allowing for the assessment of attack effectiveness in different scenarios. Additionally, we develop a comprehensive ground truth dataset specifically tailored for jailbreak prompts. This dataset is a crucial benchmark for our current study and provides a foundational resource for future research. By comparing with traditional evaluation methods, our study shows that the current results align with baseline metrics while offering a more nuanced and fine-grained assessment. It also helps identify potentially harmful attack prompts that might appear harmless in traditional evaluations. Overall, our work establishes a solid foundation for assessing a broader range of attack prompts in prompt injection.



## **18. Web Artifact Attacks Disrupt Vision Language Models**

cs.CV

**SubmitDate**: 2025-03-17    [abs](http://arxiv.org/abs/2503.13652v1) [paper-pdf](http://arxiv.org/pdf/2503.13652v1)

**Authors**: Maan Qraitem, Piotr Teterwak, Kate Saenko, Bryan A. Plummer

**Abstract**: Vision-language models (VLMs) (e.g., CLIP, LLaVA) are trained on large-scale, lightly curated web datasets, leading them to learn unintended correlations between semantic concepts and unrelated visual signals. These associations degrade model accuracy by causing predictions to rely on incidental patterns rather than genuine visual understanding. Prior work has weaponized these correlations as an attack vector to manipulate model predictions, such as inserting a deceiving class text onto the image in a typographic attack. These attacks succeed due to VLMs' text-heavy bias-a result of captions that echo visible words rather than describing content. However, this attack has focused solely on text that matches the target class exactly, overlooking a broader range of correlations, including non-matching text and graphical symbols, which arise from the abundance of branding content in web-scale data. To address this gap, we introduce artifact-based attacks: a novel class of manipulations that mislead models using both non-matching text and graphical elements. Unlike typographic attacks, these artifacts are not predefined, making them harder to defend against but also more challenging to find. We address this by framing artifact attacks as a search problem and demonstrate their effectiveness across five datasets, with some artifacts reinforcing each other to reach 100% attack success rates. These attacks transfer across models with up to 90% effectiveness, making it possible to attack unseen models. To defend against these attacks, we extend prior work's artifact aware prompting to the graphical setting. We see a moderate reduction of success rates of up to 15% relative to standard prompts, suggesting a promising direction for enhancing model robustness.



## **19. Booster: Tackling Harmful Fine-tuning for Large Language Models via Attenuating Harmful Perturbation**

cs.CL

**SubmitDate**: 2025-03-17    [abs](http://arxiv.org/abs/2409.01586v4) [paper-pdf](http://arxiv.org/pdf/2409.01586v4)

**Authors**: Tiansheng Huang, Sihao Hu, Fatih Ilhan, Selim Furkan Tekin, Ling Liu

**Abstract**: Harmful fine-tuning attack poses serious safety concerns for large language models' fine-tuning-as-a-service. While existing defenses have been proposed to mitigate the issue, their performances are still far away from satisfactory, and the root cause of the problem has not been fully recovered. To this end, we in this paper show that harmful perturbation over the model weights could be a probable cause of alignment-broken. In order to attenuate the negative impact of harmful perturbation, we propose an alignment-stage solution, dubbed Booster. Technically, along with the original alignment loss, we append a loss regularizer in the alignment stage's optimization. The regularizer ensures that the model's harmful loss reduction after the simulated harmful perturbation is attenuated, thereby mitigating the subsequent fine-tuning risk. Empirical results show that Booster can effectively reduce the harmful score of the fine-tuned models while maintaining the performance of downstream tasks. Our code is available at https://github.com/git-disl/Booster.



## **20. A Framework to Assess Multilingual Vulnerabilities of LLMs**

cs.CL

**SubmitDate**: 2025-03-17    [abs](http://arxiv.org/abs/2503.13081v1) [paper-pdf](http://arxiv.org/pdf/2503.13081v1)

**Authors**: Likai Tang, Niruth Bogahawatta, Yasod Ginige, Jiarui Xu, Shixuan Sun, Surangika Ranathunga, Suranga Seneviratne

**Abstract**: Large Language Models (LLMs) are acquiring a wider range of capabilities, including understanding and responding in multiple languages. While they undergo safety training to prevent them from answering illegal questions, imbalances in training data and human evaluation resources can make these models more susceptible to attacks in low-resource languages (LRL). This paper proposes a framework to automatically assess the multilingual vulnerabilities of commonly used LLMs. Using our framework, we evaluated six LLMs across eight languages representing varying levels of resource availability. We validated the assessments generated by our automated framework through human evaluation in two languages, demonstrating that the framework's results align with human judgments in most cases. Our findings reveal vulnerabilities in LRL; however, these may pose minimal risk as they often stem from the model's poor performance, resulting in incoherent responses.



## **21. TuBA: Cross-Lingual Transferability of Backdoor Attacks in LLMs with Instruction Tuning**

cs.CL

work in progress

**SubmitDate**: 2025-03-17    [abs](http://arxiv.org/abs/2404.19597v3) [paper-pdf](http://arxiv.org/pdf/2404.19597v3)

**Authors**: Xuanli He, Jun Wang, Qiongkai Xu, Pasquale Minervini, Pontus Stenetorp, Benjamin I. P. Rubinstein, Trevor Cohn

**Abstract**: The implications of backdoor attacks on English-centric large language models (LLMs) have been widely examined - such attacks can be achieved by embedding malicious behaviors during training and activated under specific conditions that trigger malicious outputs. Despite the increasing support for multilingual capabilities in open-source and proprietary LLMs, the impact of backdoor attacks on these systems remains largely under-explored. Our research focuses on cross-lingual backdoor attacks against multilingual LLMs, particularly investigating how poisoning the instruction-tuning data for one or two languages can affect the outputs for languages whose instruction-tuning data were not poisoned. Despite its simplicity, our empirical analysis reveals that our method exhibits remarkable efficacy in models like mT5 and GPT-4o, with high attack success rates, surpassing 90% in more than 7 out of 12 languages across various scenarios. Our findings also indicate that more powerful models show increased susceptibility to transferable cross-lingual backdoor attacks, which also applies to LLMs predominantly pre-trained on English data, such as Llama2, Llama3, and Gemma. Moreover, our experiments demonstrate 1) High Transferability: the backdoor mechanism operates successfully in cross-lingual response scenarios across 26 languages, achieving an average attack success rate of 99%, and 2) Robustness: the proposed attack remains effective even after defenses are applied. These findings expose critical security vulnerabilities in multilingual LLMs and highlight the urgent need for more robust, targeted defense strategies to address the unique challenges posed by cross-lingual backdoor transfer.



## **22. How Good is my Histopathology Vision-Language Foundation Model? A Holistic Benchmark**

eess.IV

**SubmitDate**: 2025-03-17    [abs](http://arxiv.org/abs/2503.12990v1) [paper-pdf](http://arxiv.org/pdf/2503.12990v1)

**Authors**: Roba Al Majzoub, Hashmat Malik, Muzammal Naseer, Zaigham Zaheer, Tariq Mahmood, Salman Khan, Fahad Khan

**Abstract**: Recently, histopathology vision-language foundation models (VLMs) have gained popularity due to their enhanced performance and generalizability across different downstream tasks. However, most existing histopathology benchmarks are either unimodal or limited in terms of diversity of clinical tasks, organs, and acquisition instruments, as well as their partial availability to the public due to patient data privacy. As a consequence, there is a lack of comprehensive evaluation of existing histopathology VLMs on a unified benchmark setting that better reflects a wide range of clinical scenarios. To address this gap, we introduce HistoVL, a fully open-source comprehensive benchmark comprising images acquired using up to 11 various acquisition tools that are paired with specifically crafted captions by incorporating class names and diverse pathology descriptions. Our Histo-VL includes 26 organs, 31 cancer types, and a wide variety of tissue obtained from 14 heterogeneous patient cohorts, totaling more than 5 million patches obtained from over 41K WSIs viewed under various magnification levels. We systematically evaluate existing histopathology VLMs on Histo-VL to simulate diverse tasks performed by experts in real-world clinical scenarios. Our analysis reveals interesting findings, including large sensitivity of most existing histopathology VLMs to textual changes with a drop in balanced accuracy of up to 25% in tasks such as Metastasis detection, low robustness to adversarial attacks, as well as improper calibration of models evident through high ECE values and low model prediction confidence, all of which can affect their clinical implementation.



## **23. MirrorGuard: Adaptive Defense Against Jailbreaks via Entropy-Guided Mirror Crafting**

cs.CR

**SubmitDate**: 2025-03-17    [abs](http://arxiv.org/abs/2503.12931v1) [paper-pdf](http://arxiv.org/pdf/2503.12931v1)

**Authors**: Rui Pu, Chaozhuo Li, Rui Ha, Litian Zhang, Lirong Qiu, Xi Zhang

**Abstract**: Defending large language models (LLMs) against jailbreak attacks is crucial for ensuring their safe deployment. Existing defense strategies generally rely on predefined static criteria to differentiate between harmful and benign prompts. However, such rigid rules are incapable of accommodating the inherent complexity and dynamic nature of real jailbreak attacks. In this paper, we propose a novel concept of ``mirror'' to enable dynamic and adaptive defense. A mirror refers to a dynamically generated prompt that mirrors the syntactic structure of the input while ensuring semantic safety. The personalized discrepancies between the input prompts and their corresponding mirrors serve as the guiding principles for defense. A new defense paradigm, MirrorGuard, is further proposed to detect and calibrate risky inputs based on such mirrors. An entropy-based detection metric, Relative Input Uncertainty (RIU), is integrated into MirrorGuard to quantify the discrepancies between input prompts and mirrors. MirrorGuard is evaluated on several popular datasets, demonstrating state-of-the-art defense performance while maintaining general effectiveness.



## **24. Prompt Flow Integrity to Prevent Privilege Escalation in LLM Agents**

cs.CR

**SubmitDate**: 2025-03-17    [abs](http://arxiv.org/abs/2503.15547v1) [paper-pdf](http://arxiv.org/pdf/2503.15547v1)

**Authors**: Juhee Kim, Woohyuk Choi, Byoungyoung Lee

**Abstract**: Large Language Models (LLMs) are combined with plugins to create powerful LLM agents that provide a wide range of services. Unlike traditional software, LLM agent's behavior is determined at runtime by natural language prompts from either user or plugin's data. This flexibility enables a new computing paradigm with unlimited capabilities and programmability, but also introduces new security risks, vulnerable to privilege escalation attacks. Moreover, user prompt is prone to be interpreted in an insecure way by LLM agents, creating non-deterministic behaviors that can be exploited by attackers. To address these security risks, we propose Prompt Flow Integrity (PFI), a system security-oriented solution to prevent privilege escalation in LLM agents. Analyzing the architectural characteristics of LLM agents, PFI features three mitigation techniques -- i.e., untrusted data identification, enforcing least privilege on LLM agents, and validating unsafe data flows. Our evaluation result shows that PFI effectively mitigates privilege escalation attacks while successfully preserving the utility of LLM agents.



## **25. When "Competency" in Reasoning Opens the Door to Vulnerability: Jailbreaking LLMs via Novel Complex Ciphers**

cs.CL

**SubmitDate**: 2025-03-16    [abs](http://arxiv.org/abs/2402.10601v3) [paper-pdf](http://arxiv.org/pdf/2402.10601v3)

**Authors**: Divij Handa, Zehua Zhang, Amir Saeidi, Shrinidhi Kumbhar, Chitta Baral

**Abstract**: Recent advancements in Large Language Model (LLM) safety have primarily focused on mitigating attacks crafted in natural language or common ciphers (e.g. Base64), which are likely integrated into newer models' safety training. However, we reveal a paradoxical vulnerability: as LLMs advance in reasoning, they inadvertently become more susceptible to novel jailbreaking attacks. Enhanced reasoning enables LLMs to interpret complex instructions and decode complex user-defined ciphers, creating an exploitable security gap. To study this vulnerability, we introduce Attacks using Custom Encryptions (ACE), a jailbreaking technique that encodes malicious queries with novel ciphers. Extending ACE, we introduce Layered Attacks using Custom Encryptions (LACE), which applies multi-layer ciphers to amplify attack complexity. Furthermore, we develop CipherBench, a benchmark designed to evaluate LLMs' accuracy in decoding encrypted benign text. Our experiments reveal a critical trade-off: LLMs that are more capable of decoding ciphers are more vulnerable to these jailbreaking attacks, with success rates on GPT-4o escalating from 40% under ACE to 78% with LACE. These findings highlight a critical insight: as LLMs become more adept at deciphering complex user ciphers--many of which cannot be preemptively included in safety training--they become increasingly exploitable.



## **26. h4rm3l: A language for Composable Jailbreak Attack Synthesis**

cs.CR

**SubmitDate**: 2025-03-16    [abs](http://arxiv.org/abs/2408.04811v3) [paper-pdf](http://arxiv.org/pdf/2408.04811v3)

**Authors**: Moussa Koulako Bala Doumbouya, Ananjan Nandi, Gabriel Poesia, Davide Ghilardi, Anna Goldie, Federico Bianchi, Dan Jurafsky, Christopher D. Manning

**Abstract**: Despite their demonstrated valuable capabilities, state-of-the-art (SOTA) widely deployed large language models (LLMs) still have the potential to cause harm to society due to the ineffectiveness of their safety filters, which can be bypassed by prompt transformations called jailbreak attacks. Current approaches to LLM safety assessment, which employ datasets of templated prompts and benchmarking pipelines, fail to cover sufficiently large and diverse sets of jailbreak attacks, leading to the widespread deployment of unsafe LLMs. Recent research showed that novel jailbreak attacks could be derived by composition; however, a formal composable representation for jailbreak attacks, which, among other benefits, could enable the exploration of a large compositional space of jailbreak attacks through program synthesis methods, has not been previously proposed. We introduce h4rm3l, a novel approach that addresses this gap with a human-readable domain-specific language (DSL). Our framework comprises: (1) The h4rm3l DSL, which formally expresses jailbreak attacks as compositions of parameterized string transformation primitives. (2) A synthesizer with bandit algorithms that efficiently generates jailbreak attacks optimized for a target black box LLM. (3) The h4rm3l red-teaming software toolkit that employs the previous two components and an automated harmful LLM behavior classifier that is strongly aligned with human judgment. We demonstrate h4rm3l's efficacy by synthesizing a dataset of 2656 successful novel jailbreak attacks targeting 6 SOTA open-source and proprietary LLMs, and by benchmarking those models against a subset of these synthesized attacks. Our results show that h4rm3l's synthesized attacks are diverse and more successful than existing jailbreak attacks in literature, with success rates exceeding 90% on SOTA LLMs.



## **27. Systematic Categorization, Construction and Evaluation of New Attacks against Multi-modal Mobile GUI Agents**

cs.CR

Preprint. Work in progress

**SubmitDate**: 2025-03-16    [abs](http://arxiv.org/abs/2407.09295v3) [paper-pdf](http://arxiv.org/pdf/2407.09295v3)

**Authors**: Yulong Yang, Xinshan Yang, Shuaidong Li, Chenhao Lin, Zhengyu Zhao, Chao Shen, Tianwei Zhang

**Abstract**: The integration of Large Language Models (LLMs) and Multi-modal Large Language Models (MLLMs) into mobile GUI agents has significantly enhanced user efficiency and experience. However, this advancement also introduces potential security vulnerabilities that have yet to be thoroughly explored. In this paper, we present a systematic security investigation of multi-modal mobile GUI agents, addressing this critical gap in the existing literature. Our contributions are twofold: (1) we propose a novel threat modeling methodology, leading to the discovery and feasibility analysis of 34 previously unreported attacks, and (2) we design an attack framework to systematically construct and evaluate these threats. Through a combination of real-world case studies and extensive dataset-driven experiments, we validate the severity and practicality of those attacks, highlighting the pressing need for robust security measures in mobile GUI systems.



## **28. One Goal, Many Challenges: Robust Preference Optimization Amid Content-Aware and Multi-Source Noise**

cs.LG

**SubmitDate**: 2025-03-16    [abs](http://arxiv.org/abs/2503.12301v1) [paper-pdf](http://arxiv.org/pdf/2503.12301v1)

**Authors**: Amirabbas Afzali, Amirhossein Afsharrad, Seyed Shahabeddin Mousavi, Sanjay Lall

**Abstract**: Large Language Models (LLMs) have made significant strides in generating human-like responses, largely due to preference alignment techniques. However, these methods often assume unbiased human feedback, which is rarely the case in real-world scenarios. This paper introduces Content-Aware Noise-Resilient Preference Optimization (CNRPO), a novel framework that addresses multiple sources of content-dependent noise in preference learning. CNRPO employs a multi-objective optimization approach to separate true preferences from content-aware noises, effectively mitigating their impact. We leverage backdoor attack mechanisms to efficiently learn and control various noise sources within a single model. Theoretical analysis and extensive experiments on different synthetic noisy datasets demonstrate that CNRPO significantly improves alignment with primary human preferences while controlling for secondary noises and biases, such as response length and harmfulness.



## **29. From ML to LLM: Evaluating the Robustness of Phishing Webpage Detection Models against Adversarial Attacks**

cs.CR

**SubmitDate**: 2025-03-15    [abs](http://arxiv.org/abs/2407.20361v3) [paper-pdf](http://arxiv.org/pdf/2407.20361v3)

**Authors**: Aditya Kulkarni, Vivek Balachandran, Dinil Mon Divakaran, Tamal Das

**Abstract**: Phishing attacks attempt to deceive users into stealing sensitive information, posing a significant cybersecurity threat. Advances in machine learning (ML) and deep learning (DL) have led to the development of numerous phishing webpage detection solutions, but these models remain vulnerable to adversarial attacks. Evaluating their robustness against adversarial phishing webpages is essential. Existing tools contain datasets of pre-designed phishing webpages for a limited number of brands, and lack diversity in phishing features.   To address these challenges, we develop PhishOracle, a tool that generates adversarial phishing webpages by embedding diverse phishing features into legitimate webpages. We evaluate the robustness of three existing task-specific models -- Stack model, VisualPhishNet, and Phishpedia -- against PhishOracle-generated adversarial phishing webpages and observe a significant drop in their detection rates. In contrast, a multimodal large language model (MLLM)-based phishing detector demonstrates stronger robustness against these adversarial attacks but still is prone to evasion. Our findings highlight the vulnerability of phishing detection models to adversarial attacks, emphasizing the need for more robust detection approaches. Furthermore, we conduct a user study to evaluate whether PhishOracle-generated adversarial phishing webpages can deceive users. The results show that many of these phishing webpages evade not only existing detection models but also users. We also develop the PhishOracle web app, allowing users to input a legitimate URL, select relevant phishing features and generate a corresponding phishing webpage. All resources will be made publicly available on GitHub.



## **30. JailGuard: A Universal Detection Framework for LLM Prompt-based Attacks**

cs.CR

40 pages, 12 figures

**SubmitDate**: 2025-03-15    [abs](http://arxiv.org/abs/2312.10766v4) [paper-pdf](http://arxiv.org/pdf/2312.10766v4)

**Authors**: Xiaoyu Zhang, Cen Zhang, Tianlin Li, Yihao Huang, Xiaojun Jia, Ming Hu, Jie Zhang, Yang Liu, Shiqing Ma, Chao Shen

**Abstract**: The systems and software powered by Large Language Models (LLMs) and Multi-Modal LLMs (MLLMs) have played a critical role in numerous scenarios. However, current LLM systems are vulnerable to prompt-based attacks, with jailbreaking attacks enabling the LLM system to generate harmful content, while hijacking attacks manipulate the LLM system to perform attacker-desired tasks, underscoring the necessity for detection tools. Unfortunately, existing detecting approaches are usually tailored to specific attacks, resulting in poor generalization in detecting various attacks across different modalities. To address it, we propose JailGuard, a universal detection framework deployed on top of LLM systems for prompt-based attacks across text and image modalities. JailGuard operates on the principle that attacks are inherently less robust than benign ones. Specifically, JailGuard mutates untrusted inputs to generate variants and leverages the discrepancy of the variants' responses on the target model to distinguish attack samples from benign samples. We implement 18 mutators for text and image inputs and design a mutator combination policy to further improve detection generalization. The evaluation on the dataset containing 15 known attack types suggests that JailGuard achieves the best detection accuracy of 86.14%/82.90% on text and image inputs, outperforming state-of-the-art methods by 11.81%-25.73% and 12.20%-21.40%.



## **31. Making Every Step Effective: Jailbreaking Large Vision-Language Models Through Hierarchical KV Equalization**

cs.CV

**SubmitDate**: 2025-03-14    [abs](http://arxiv.org/abs/2503.11750v1) [paper-pdf](http://arxiv.org/pdf/2503.11750v1)

**Authors**: Shuyang Hao, Yiwei Wang, Bryan Hooi, Jun Liu, Muhao Chen, Zi Huang, Yujun Cai

**Abstract**: In the realm of large vision-language models (LVLMs), adversarial jailbreak attacks serve as a red-teaming approach to identify safety vulnerabilities of these models and their associated defense mechanisms. However, we identify a critical limitation: not every adversarial optimization step leads to a positive outcome, and indiscriminately accepting optimization results at each step may reduce the overall attack success rate. To address this challenge, we introduce HKVE (Hierarchical Key-Value Equalization), an innovative jailbreaking framework that selectively accepts gradient optimization results based on the distribution of attention scores across different layers, ensuring that every optimization step positively contributes to the attack. Extensive experiments demonstrate HKVE's significant effectiveness, achieving attack success rates of 75.08% on MiniGPT4, 85.84% on LLaVA and 81.00% on Qwen-VL, substantially outperforming existing methods by margins of 20.43\%, 21.01\% and 26.43\% respectively. Furthermore, making every step effective not only leads to an increase in attack success rate but also allows for a reduction in the number of iterations, thereby lowering computational costs. Warning: This paper contains potentially harmful example data.



## **32. Tit-for-Tat: Safeguarding Large Vision-Language Models Against Jailbreak Attacks via Adversarial Defense**

cs.CR

**SubmitDate**: 2025-03-14    [abs](http://arxiv.org/abs/2503.11619v1) [paper-pdf](http://arxiv.org/pdf/2503.11619v1)

**Authors**: Shuyang Hao, Yiwei Wang, Bryan Hooi, Ming-Hsuan Yang, Jun Liu, Chengcheng Tang, Zi Huang, Yujun Cai

**Abstract**: Deploying large vision-language models (LVLMs) introduces a unique vulnerability: susceptibility to malicious attacks via visual inputs. However, existing defense methods suffer from two key limitations: (1) They solely focus on textual defenses, fail to directly address threats in the visual domain where attacks originate, and (2) the additional processing steps often incur significant computational overhead or compromise model performance on benign tasks. Building on these insights, we propose ESIII (Embedding Security Instructions Into Images), a novel methodology for transforming the visual space from a source of vulnerability into an active defense mechanism. Initially, we embed security instructions into defensive images through gradient-based optimization, obtaining security instructions in the visual dimension. Subsequently, we integrate security instructions from visual and textual dimensions with the input query. The collaboration between security instructions from different dimensions ensures comprehensive security protection. Extensive experiments demonstrate that our approach effectively fortifies the robustness of LVLMs against such attacks while preserving their performance on standard benign tasks and incurring an imperceptible increase in time costs.



## **33. Align in Depth: Defending Jailbreak Attacks via Progressive Answer Detoxification**

cs.CR

**SubmitDate**: 2025-03-14    [abs](http://arxiv.org/abs/2503.11185v1) [paper-pdf](http://arxiv.org/pdf/2503.11185v1)

**Authors**: Yingjie Zhang, Tong Liu, Zhe Zhao, Guozhu Meng, Kai Chen

**Abstract**: Large Language Models (LLMs) are vulnerable to jailbreak attacks, which use crafted prompts to elicit toxic responses. These attacks exploit LLMs' difficulty in dynamically detecting harmful intents during the generation process. Traditional safety alignment methods, often relying on the initial few generation steps, are ineffective due to limited computational budget. This paper proposes DEEPALIGN, a robust defense framework that fine-tunes LLMs to progressively detoxify generated content, significantly improving both the computational budget and effectiveness of mitigating harmful generation. Our approach uses a hybrid loss function operating on hidden states to directly improve LLMs' inherent awareness of toxity during generation. Furthermore, we redefine safe responses by generating semantically relevant answers to harmful queries, thereby increasing robustness against representation-mutation attacks. Evaluations across multiple LLMs demonstrate state-of-the-art defense performance against six different attack types, reducing Attack Success Rates by up to two orders of magnitude compared to previous state-of-the-art defense while preserving utility. This work advances LLM safety by addressing limitations of conventional alignment through dynamic, context-aware mitigation.



## **34. Towards a Systematic Evaluation of Hallucinations in Large-Vision Language Models**

cs.CV

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2412.20622v2) [paper-pdf](http://arxiv.org/pdf/2412.20622v2)

**Authors**: Ashish Seth, Dinesh Manocha, Chirag Agarwal

**Abstract**: Large Vision-Language Models (LVLMs) have demonstrated remarkable performance in complex multimodal tasks. However, these models still suffer from hallucinations, particularly when required to implicitly recognize or infer diverse visual entities from images for complex vision-language tasks. To address this challenge, we propose HALLUCINOGEN, a novel visual question answering (VQA) benchmark that employs contextual reasoning prompts as hallucination attacks to evaluate the extent of hallucination in state-of-the-art LVLMs. Our benchmark provides a comprehensive study of the implicit reasoning capabilities of these models by first categorizing visual entities based on the ease of recognition in an image as either salient (prominent, visibly recognizable objects such as a car) or latent entities (such as identifying a disease from a chest X-ray), which are not readily visible and require domain knowledge or contextual reasoning for accurate inference. Next, we design hallucination attacks for both types of entities to assess hallucinations in LVLMs while performing various vision-language tasks, such as locating or reasoning about specific entities within an image, where models must perform implicit reasoning by verifying the existence of the queried entity within the image before generating responses. Finally, our extensive evaluations of eleven LVLMs, including powerful open-source models (like LLaMA-3.2 and DeepSeek-V2), commercial models like Gemini, and two hallucination mitigation strategies across multiple datasets, demonstrate that current LVLMs remain susceptible to hallucination attacks.



## **35. ChatGPT Encounters Morphing Attack Detection: Zero-Shot MAD with Multi-Modal Large Language Models and General Vision Models**

cs.CV

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.10937v1) [paper-pdf](http://arxiv.org/pdf/2503.10937v1)

**Authors**: Haoyu Zhang, Raghavendra Ramachandra, Kiran Raja, Christoph Busch

**Abstract**: Face Recognition Systems (FRS) are increasingly vulnerable to face-morphing attacks, prompting the development of Morphing Attack Detection (MAD) algorithms. However, a key challenge in MAD lies in its limited generalizability to unseen data and its lack of explainability-critical for practical application environments such as enrolment stations and automated border control systems. Recognizing that most existing MAD algorithms rely on supervised learning paradigms, this work explores a novel approach to MAD using zero-shot learning leveraged on Large Language Models (LLMs). We propose two types of zero-shot MAD algorithms: one leveraging general vision models and the other utilizing multimodal LLMs. For general vision models, we address the MAD task by computing the mean support embedding of an independent support set without using morphed images. For the LLM-based approach, we employ the state-of-the-art GPT-4 Turbo API with carefully crafted prompts. To evaluate the feasibility of zero-shot MAD and the effectiveness of the proposed methods, we constructed a print-scan morph dataset featuring various unseen morphing algorithms, simulating challenging real-world application scenarios. Experimental results demonstrated notable detection accuracy, validating the applicability of zero-shot learning for MAD tasks. Additionally, our investigation into LLM-based MAD revealed that multimodal LLMs, such as ChatGPT, exhibit remarkable generalizability to untrained MAD tasks. Furthermore, they possess a unique ability to provide explanations and guidance, which can enhance transparency and usability for end-users in practical applications.



## **36. A Frustratingly Simple Yet Highly Effective Attack Baseline: Over 90% Success Rate Against the Strong Black-box Models of GPT-4.5/4o/o1**

cs.CV

Code at: https://github.com/VILA-Lab/M-Attack

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.10635v1) [paper-pdf](http://arxiv.org/pdf/2503.10635v1)

**Authors**: Zhaoyi Li, Xiaohan Zhao, Dong-Dong Wu, Jiacheng Cui, Zhiqiang Shen

**Abstract**: Despite promising performance on open-source large vision-language models (LVLMs), transfer-based targeted attacks often fail against black-box commercial LVLMs. Analyzing failed adversarial perturbations reveals that the learned perturbations typically originate from a uniform distribution and lack clear semantic details, resulting in unintended responses. This critical absence of semantic information leads commercial LVLMs to either ignore the perturbation entirely or misinterpret its embedded semantics, thereby causing the attack to fail. To overcome these issues, we notice that identifying core semantic objects is a key objective for models trained with various datasets and methodologies. This insight motivates our approach that refines semantic clarity by encoding explicit semantic details within local regions, thus ensuring interoperability and capturing finer-grained features, and by concentrating modifications on semantically rich areas rather than applying them uniformly. To achieve this, we propose a simple yet highly effective solution: at each optimization step, the adversarial image is cropped randomly by a controlled aspect ratio and scale, resized, and then aligned with the target image in the embedding space. Experimental results confirm our hypothesis. Our adversarial examples crafted with local-aggregated perturbations focused on crucial regions exhibit surprisingly good transferability to commercial LVLMs, including GPT-4.5, GPT-4o, Gemini-2.0-flash, Claude-3.5-sonnet, Claude-3.7-sonnet, and even reasoning models like o1, Claude-3.7-thinking and Gemini-2.0-flash-thinking. Our approach achieves success rates exceeding 90% on GPT-4.5, 4o, and o1, significantly outperforming all prior state-of-the-art attack methods. Our optimized adversarial examples under different configurations and training code are available at https://github.com/VILA-Lab/M-Attack.



## **37. ASIDE: Architectural Separation of Instructions and Data in Language Models**

cs.LG

ICLR 2025 Workshop on Building Trust in Language Models and  Applications

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.10566v1) [paper-pdf](http://arxiv.org/pdf/2503.10566v1)

**Authors**: Egor Zverev, Evgenii Kortukov, Alexander Panfilov, Soroush Tabesh, Alexandra Volkova, Sebastian Lapuschkin, Wojciech Samek, Christoph H. Lampert

**Abstract**: Despite their remarkable performance, large language models lack elementary safety features, and this makes them susceptible to numerous malicious attacks. In particular, previous work has identified the absence of an intrinsic separation between instructions and data as a root cause for the success of prompt injection attacks. In this work, we propose an architectural change, ASIDE, that allows the model to clearly separate between instructions and data by using separate embeddings for them. Instead of training the embeddings from scratch, we propose a method to convert an existing model to ASIDE form by using two copies of the original model's embeddings layer, and applying an orthogonal rotation to one of them. We demonstrate the effectiveness of our method by showing (1) highly increased instruction-data separation scores without a loss in model capabilities and (2) competitive results on prompt injection benchmarks, even without dedicated safety training. Additionally, we study the working mechanism behind our method through an analysis of model representations.



## **38. TH-Bench: Evaluating Evading Attacks via Humanizing AI Text on Machine-Generated Text Detectors**

cs.CR

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.08708v2) [paper-pdf](http://arxiv.org/pdf/2503.08708v2)

**Authors**: Jingyi Zheng, Junfeng Wang, Zhen Sun, Wenhan Dong, Yule Liu, Xinlei He

**Abstract**: As Large Language Models (LLMs) advance, Machine-Generated Texts (MGTs) have become increasingly fluent, high-quality, and informative. Existing wide-range MGT detectors are designed to identify MGTs to prevent the spread of plagiarism and misinformation. However, adversaries attempt to humanize MGTs to evade detection (named evading attacks), which requires only minor modifications to bypass MGT detectors. Unfortunately, existing attacks generally lack a unified and comprehensive evaluation framework, as they are assessed using different experimental settings, model architectures, and datasets. To fill this gap, we introduce the Text-Humanization Benchmark (TH-Bench), the first comprehensive benchmark to evaluate evading attacks against MGT detectors. TH-Bench evaluates attacks across three key dimensions: evading effectiveness, text quality, and computational overhead. Our extensive experiments evaluate 6 state-of-the-art attacks against 13 MGT detectors across 6 datasets, spanning 19 domains and generated by 11 widely used LLMs. Our findings reveal that no single evading attack excels across all three dimensions. Through in-depth analysis, we highlight the strengths and limitations of different attacks. More importantly, we identify a trade-off among three dimensions and propose two optimization insights. Through preliminary experiments, we validate their correctness and effectiveness, offering potential directions for future research.



## **39. Prompt Inversion Attack against Collaborative Inference of Large Language Models**

cs.CR

To appear at IEEE Symposium on Security and Privacy 2025

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.09022v2) [paper-pdf](http://arxiv.org/pdf/2503.09022v2)

**Authors**: Wenjie Qu, Yuguang Zhou, Yongji Wu, Tingsong Xiao, Binhang Yuan, Yiming Li, Jiaheng Zhang

**Abstract**: Large language models (LLMs) have been widely applied for their remarkable capability of content generation. However, the practical use of open-source LLMs is hindered by high resource requirements, making deployment expensive and limiting widespread development. The collaborative inference is a promising solution for this problem, in which users collaborate by each hosting a subset of layers and transmitting intermediate activation. Many companies are building collaborative inference platforms to reduce LLM serving costs, leveraging users' underutilized GPUs. Despite widespread interest in collaborative inference within academia and industry, the privacy risks associated with LLM collaborative inference have not been well studied. This is largely because of the challenge posed by inverting LLM activation due to its strong non-linearity.   In this paper, to validate the severity of privacy threats in LLM collaborative inference, we introduce the concept of prompt inversion attack (PIA), where a malicious participant intends to recover the input prompt through the activation transmitted by its previous participant. Extensive experiments show that our PIA method substantially outperforms existing baselines. For example, our method achieves an 88.4\% token accuracy on the Skytrax dataset with the Llama-65B model when inverting the maximum number of transformer layers, while the best baseline method only achieves 22.8\% accuracy. The results verify the effectiveness of our PIA attack and highlights its practical threat to LLM collaborative inference systems.



## **40. ExtremeAIGC: Benchmarking LMM Vulnerability to AI-Generated Extremist Content**

cs.CR

Preprint

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.09964v1) [paper-pdf](http://arxiv.org/pdf/2503.09964v1)

**Authors**: Bhavik Chandna, Mariam Aboujenane, Usman Naseem

**Abstract**: Large Multimodal Models (LMMs) are increasingly vulnerable to AI-generated extremist content, including photorealistic images and text, which can be used to bypass safety mechanisms and generate harmful outputs. However, existing datasets for evaluating LMM robustness offer limited exploration of extremist content, often lacking AI-generated images, diverse image generation models, and comprehensive coverage of historical events, which hinders a complete assessment of model vulnerabilities. To fill this gap, we introduce ExtremeAIGC, a benchmark dataset and evaluation framework designed to assess LMM vulnerabilities against such content. ExtremeAIGC simulates real-world events and malicious use cases by curating diverse text- and image-based examples crafted using state-of-the-art image generation techniques. Our study reveals alarming weaknesses in LMMs, demonstrating that even cutting-edge safety measures fail to prevent the generation of extremist material. We systematically quantify the success rates of various attack strategies, exposing critical gaps in current defenses and emphasizing the need for more robust mitigation strategies.



## **41. Adversarial Vulnerabilities in Large Language Models for Time Series Forecasting**

cs.LG

AISTATS 2025

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2412.08099v4) [paper-pdf](http://arxiv.org/pdf/2412.08099v4)

**Authors**: Fuqiang Liu, Sicong Jiang, Luis Miranda-Moreno, Seongjin Choi, Lijun Sun

**Abstract**: Large Language Models (LLMs) have recently demonstrated significant potential in time series forecasting, offering impressive capabilities in handling complex temporal data. However, their robustness and reliability in real-world applications remain under-explored, particularly concerning their susceptibility to adversarial attacks. In this paper, we introduce a targeted adversarial attack framework for LLM-based time series forecasting. By employing both gradient-free and black-box optimization methods, we generate minimal yet highly effective perturbations that significantly degrade the forecasting accuracy across multiple datasets and LLM architectures. Our experiments, which include models like LLMTime with GPT-3.5, GPT-4, LLaMa, and Mistral, TimeGPT, and TimeLLM show that adversarial attacks lead to much more severe performance degradation than random noise, and demonstrate the broad effectiveness of our attacks across different LLMs. The results underscore the critical vulnerabilities of LLMs in time series forecasting, highlighting the need for robust defense mechanisms to ensure their reliable deployment in practical applications. The code repository can be found at https://github.com/JohnsonJiang1996/AdvAttack_LLM4TS.



## **42. CyberLLMInstruct: A New Dataset for Analysing Safety of Fine-Tuned LLMs Using Cyber Security Data**

cs.CR

The paper is submitted to "The 48th International ACM SIGIR  Conference on Research and Development in Information Retrieval" and is  currently under review

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.09334v1) [paper-pdf](http://arxiv.org/pdf/2503.09334v1)

**Authors**: Adel ElZemity, Budi Arief, Shujun Li

**Abstract**: The integration of large language models (LLMs) into cyber security applications presents significant opportunities, such as enhancing threat analysis and malware detection, but can also introduce critical risks and safety concerns, including personal data leakage and automated generation of new malware. To address these challenges, we developed CyberLLMInstruct, a dataset of 54,928 instruction-response pairs spanning cyber security tasks such as malware analysis, phishing simulations, and zero-day vulnerabilities. The dataset was constructed through a multi-stage process. This involved sourcing data from multiple resources, filtering and structuring it into instruction-response pairs, and aligning it with real-world scenarios to enhance its applicability. Seven open-source LLMs were chosen to test the usefulness of CyberLLMInstruct: Phi 3 Mini 3.8B, Mistral 7B, Qwen 2.5 7B, Llama 3 8B, Llama 3.1 8B, Gemma 2 9B, and Llama 2 70B. In our primary example, we rigorously assess the safety of fine-tuned models using the OWASP top 10 framework, finding that fine-tuning reduces safety resilience across all tested LLMs and every adversarial attack (e.g., the security score of Llama 3.1 8B against prompt injection drops from 0.95 to 0.15). In our second example, we show that these same fine-tuned models can also achieve up to 92.50 percent accuracy on the CyberMetric benchmark. These findings highlight a trade-off between performance and safety, showing the importance of adversarial testing and further research into fine-tuning methodologies that can mitigate safety risks while still improving performance across diverse datasets and domains. All scripts required to reproduce the dataset, along with examples and relevant resources for replicating our results, will be made available upon the paper's acceptance.



## **43. Prompt Inference Attack on Distributed Large Language Model Inference Frameworks**

cs.CR

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.09291v1) [paper-pdf](http://arxiv.org/pdf/2503.09291v1)

**Authors**: Xinjian Luo, Ting Yu, Xiaokui Xiao

**Abstract**: The inference process of modern large language models (LLMs) demands prohibitive computational resources, rendering them infeasible for deployment on consumer-grade devices. To address this limitation, recent studies propose distributed LLM inference frameworks, which employ split learning principles to enable collaborative LLM inference on resource-constrained hardware. However, distributing LLM layers across participants requires the transmission of intermediate outputs, which may introduce privacy risks to the original input prompts - a critical issue that has yet to be thoroughly explored in the literature.   In this paper, we rigorously examine the privacy vulnerabilities of distributed LLM inference frameworks by designing and evaluating three prompt inference attacks aimed at reconstructing input prompts from intermediate LLM outputs. These attacks are developed under various query and data constraints to reflect diverse real-world LLM service scenarios. Specifically, the first attack assumes an unlimited query budget and access to an auxiliary dataset sharing the same distribution as the target prompts. The second attack also leverages unlimited queries but uses an auxiliary dataset with a distribution differing from the target prompts. The third attack operates under the most restrictive scenario, with limited query budgets and no auxiliary dataset available. We evaluate these attacks on a range of LLMs, including state-of-the-art models such as Llama-3.2 and Phi-3.5, as well as widely-used models like GPT-2 and BERT for comparative analysis. Our experiments show that the first two attacks achieve reconstruction accuracies exceeding 90%, while the third achieves accuracies typically above 50%, even under stringent constraints. These findings highlight privacy risks in distributed LLM inference frameworks, issuing a strong alert on their deployment in real-world applications.



## **44. In-Context Defense in Computer Agents: An Empirical Study**

cs.AI

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.09241v1) [paper-pdf](http://arxiv.org/pdf/2503.09241v1)

**Authors**: Pei Yang, Hai Ci, Mike Zheng Shou

**Abstract**: Computer agents powered by vision-language models (VLMs) have significantly advanced human-computer interaction, enabling users to perform complex tasks through natural language instructions. However, these agents are vulnerable to context deception attacks, an emerging threat where adversaries embed misleading content into the agent's operational environment, such as a pop-up window containing deceptive instructions. Existing defenses, such as instructing agents to ignore deceptive elements, have proven largely ineffective. As the first systematic study on protecting computer agents, we introduce textbf{in-context defense}, leveraging in-context learning and chain-of-thought (CoT) reasoning to counter such attacks. Our approach involves augmenting the agent's context with a small set of carefully curated exemplars containing both malicious environments and corresponding defensive responses. These exemplars guide the agent to first perform explicit defensive reasoning before action planning, reducing susceptibility to deceptive attacks. Experiments demonstrate the effectiveness of our method, reducing attack success rates by 91.2% on pop-up window attacks, 74.6% on average on environment injection attacks, while achieving 100% successful defenses against distracting advertisements. Our findings highlight that (1) defensive reasoning must precede action planning for optimal performance, and (2) a minimal number of exemplars (fewer than three) is sufficient to induce an agent's defensive behavior.



## **45. DetectRL: Benchmarking LLM-Generated Text Detection in Real-World Scenarios**

cs.CL

Accepted to NeurIPS 2024 Datasets and Benchmarks Track (Camera-Ready)

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2410.23746v3) [paper-pdf](http://arxiv.org/pdf/2410.23746v3)

**Authors**: Junchao Wu, Runzhe Zhan, Derek F. Wong, Shu Yang, Xinyi Yang, Yulin Yuan, Lidia S. Chao

**Abstract**: Detecting text generated by large language models (LLMs) is of great recent interest. With zero-shot methods like DetectGPT, detection capabilities have reached impressive levels. However, the reliability of existing detectors in real-world applications remains underexplored. In this study, we present a new benchmark, DetectRL, highlighting that even state-of-the-art (SOTA) detection techniques still underperformed in this task. We collected human-written datasets from domains where LLMs are particularly prone to misuse. Using popular LLMs, we generated data that better aligns with real-world applications. Unlike previous studies, we employed heuristic rules to create adversarial LLM-generated text, simulating various prompts usages, human revisions like word substitutions, and writing noises like spelling mistakes. Our development of DetectRL reveals the strengths and limitations of current SOTA detectors. More importantly, we analyzed the potential impact of writing styles, model types, attack methods, the text lengths, and real-world human writing factors on different types of detectors. We believe DetectRL could serve as an effective benchmark for assessing detectors in real-world scenarios, evolving with advanced attack methods, thus providing more stressful evaluation to drive the development of more efficient detectors. Data and code are publicly available at: https://github.com/NLP2CT/DetectRL.



## **46. A Survey on Trustworthy LLM Agents: Threats and Countermeasures**

cs.MA

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.09648v1) [paper-pdf](http://arxiv.org/pdf/2503.09648v1)

**Authors**: Miao Yu, Fanci Meng, Xinyun Zhou, Shilong Wang, Junyuan Mao, Linsey Pang, Tianlong Chen, Kun Wang, Xinfeng Li, Yongfeng Zhang, Bo An, Qingsong Wen

**Abstract**: With the rapid evolution of Large Language Models (LLMs), LLM-based agents and Multi-agent Systems (MAS) have significantly expanded the capabilities of LLM ecosystems. This evolution stems from empowering LLMs with additional modules such as memory, tools, environment, and even other agents. However, this advancement has also introduced more complex issues of trustworthiness, which previous research focused solely on LLMs could not cover. In this survey, we propose the TrustAgent framework, a comprehensive study on the trustworthiness of agents, characterized by modular taxonomy, multi-dimensional connotations, and technical implementation. By thoroughly investigating and summarizing newly emerged attacks, defenses, and evaluation methods for agents and MAS, we extend the concept of Trustworthy LLM to the emerging paradigm of Trustworthy Agent. In TrustAgent, we begin by deconstructing and introducing various components of the Agent and MAS. Then, we categorize their trustworthiness into intrinsic (brain, memory, and tool) and extrinsic (user, agent, and environment) aspects. Subsequently, we delineate the multifaceted meanings of trustworthiness and elaborate on the implementation techniques of existing research related to these internal and external modules. Finally, we present our insights and outlook on this domain, aiming to provide guidance for future endeavors.



## **47. Probing Latent Subspaces in LLM for AI Security: Identifying and Manipulating Adversarial States**

cs.LG

4 figures

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.09066v1) [paper-pdf](http://arxiv.org/pdf/2503.09066v1)

**Authors**: Xin Wei Chia, Jonathan Pan

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across various tasks, yet they remain vulnerable to adversarial manipulations such as jailbreaking via prompt injection attacks. These attacks bypass safety mechanisms to generate restricted or harmful content. In this study, we investigated the underlying latent subspaces of safe and jailbroken states by extracting hidden activations from a LLM. Inspired by attractor dynamics in neuroscience, we hypothesized that LLM activations settle into semi stable states that can be identified and perturbed to induce state transitions. Using dimensionality reduction techniques, we projected activations from safe and jailbroken responses to reveal latent subspaces in lower dimensional spaces. We then derived a perturbation vector that when applied to safe representations, shifted the model towards a jailbreak state. Our results demonstrate that this causal intervention results in statistically significant jailbreak responses in a subset of prompts. Next, we probed how these perturbations propagate through the model's layers, testing whether the induced state change remains localized or cascades throughout the network. Our findings indicate that targeted perturbations induced distinct shifts in activations and model responses. Our approach paves the way for potential proactive defenses, shifting from traditional guardrail based methods to preemptive, model agnostic techniques that neutralize adversarial states at the representation level.



## **48. Battling Misinformation: An Empirical Study on Adversarial Factuality in Open-Source Large Language Models**

cs.CL

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.10690v1) [paper-pdf](http://arxiv.org/pdf/2503.10690v1)

**Authors**: Shahnewaz Karim Sakib, Anindya Bijoy Das, Shibbir Ahmed

**Abstract**: Adversarial factuality refers to the deliberate insertion of misinformation into input prompts by an adversary, characterized by varying levels of expressed confidence. In this study, we systematically evaluate the performance of several open-source large language models (LLMs) when exposed to such adversarial inputs. Three tiers of adversarial confidence are considered: strongly confident, moderately confident, and limited confidence. Our analysis encompasses eight LLMs: LLaMA 3.1 (8B), Phi 3 (3.8B), Qwen 2.5 (7B), Deepseek-v2 (16B), Gemma2 (9B), Falcon (7B), Mistrallite (7B), and LLaVA (7B). Empirical results indicate that LLaMA 3.1 (8B) exhibits a robust capability in detecting adversarial inputs, whereas Falcon (7B) shows comparatively lower performance. Notably, for the majority of the models, detection success improves as the adversary's confidence decreases; however, this trend is reversed for LLaMA 3.1 (8B) and Phi 3 (3.8B), where a reduction in adversarial confidence corresponds with diminished detection performance. Further analysis of the queries that elicited the highest and lowest rates of successful attacks reveals that adversarial attacks are more effective when targeting less commonly referenced or obscure information.



## **49. JBFuzz: Jailbreaking LLMs Efficiently and Effectively Using Fuzzing**

cs.CR

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.08990v1) [paper-pdf](http://arxiv.org/pdf/2503.08990v1)

**Authors**: Vasudev Gohil

**Abstract**: Large language models (LLMs) have shown great promise as language understanding and decision making tools, and they have permeated various aspects of our everyday life. However, their widespread availability also comes with novel risks, such as generating harmful, unethical, or offensive content, via an attack called jailbreaking. Despite extensive efforts from LLM developers to align LLMs using human feedback, they are still susceptible to jailbreak attacks. To tackle this issue, researchers often employ red-teaming to understand and investigate jailbreak prompts. However, existing red-teaming approaches lack effectiveness, scalability, or both. To address these issues, we propose JBFuzz, a novel effective, automated, and scalable red-teaming technique for jailbreaking LLMs.   JBFuzz is inspired by the success of fuzzing for detecting bugs/vulnerabilities in software. We overcome three challenges related to effectiveness and scalability by devising novel seed prompts, a lightweight mutation engine, and a lightweight and accurate evaluator for guiding the fuzzer. Assimilating all three solutions results in a potent fuzzer that only requires black-box access to the target LLM. We perform extensive experimental evaluation of JBFuzz using nine popular and widely-used LLMs. We find that JBFuzz successfully jailbreaks all LLMs for various harmful/unethical questions, with an average attack success rate of 99%. We also find that JBFuzz is extremely efficient as it jailbreaks a given LLM for a given question in 60 seconds on average. Our work highlights the susceptibility of the state-of-the-art LLMs to jailbreak attacks even after safety alignment, and serves as a valuable red-teaming tool for LLM developers.



## **50. Iterative Self-Tuning LLMs for Enhanced Jailbreaking Capabilities**

cs.CL

Accepted to NAACL 2025 Main (oral)

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2410.18469v3) [paper-pdf](http://arxiv.org/pdf/2410.18469v3)

**Authors**: Chung-En Sun, Xiaodong Liu, Weiwei Yang, Tsui-Wei Weng, Hao Cheng, Aidan San, Michel Galley, Jianfeng Gao

**Abstract**: Recent research has shown that Large Language Models (LLMs) are vulnerable to automated jailbreak attacks, where adversarial suffixes crafted by algorithms appended to harmful queries bypass safety alignment and trigger unintended responses. Current methods for generating these suffixes are computationally expensive and have low Attack Success Rates (ASR), especially against well-aligned models like Llama2 and Llama3. To overcome these limitations, we introduce ADV-LLM, an iterative self-tuning process that crafts adversarial LLMs with enhanced jailbreak ability. Our framework significantly reduces the computational cost of generating adversarial suffixes while achieving nearly 100\% ASR on various open-source LLMs. Moreover, it exhibits strong attack transferability to closed-source models, achieving 99\% ASR on GPT-3.5 and 49\% ASR on GPT-4, despite being optimized solely on Llama3. Beyond improving jailbreak ability, ADV-LLM provides valuable insights for future safety alignment research through its ability to generate large datasets for studying LLM safety.



