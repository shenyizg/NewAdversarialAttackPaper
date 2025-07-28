# Latest Large Language Model Attack Papers
**update at 2025-07-28 19:12:42**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. Jailbreaking Large Language Diffusion Models: Revealing Hidden Safety Flaws in Diffusion-Based Text Generation**

cs.CL

**SubmitDate**: 2025-07-25    [abs](http://arxiv.org/abs/2507.19227v1) [paper-pdf](http://arxiv.org/pdf/2507.19227v1)

**Authors**: Yuanhe Zhang, Fangzhou Xie, Zhenhong Zhou, Zherui Li, Hao Chen, Kun Wang, Yufei Guo

**Abstract**: Large Language Diffusion Models (LLDMs) exhibit comparable performance to LLMs while offering distinct advantages in inference speed and mathematical reasoning tasks.The precise and rapid generation capabilities of LLDMs amplify concerns of harmful generations, while existing jailbreak methodologies designed for Large Language Models (LLMs) prove limited effectiveness against LLDMs and fail to expose safety vulnerabilities.Successful defense cannot definitively resolve harmful generation concerns, as it remains unclear whether LLDMs possess safety robustness or existing attacks are incompatible with diffusion-based architectures.To address this, we first reveal the vulnerability of LLDMs to jailbreak and demonstrate that attack failure in LLDMs stems from fundamental architectural differences.We present a PArallel Decoding jailbreak (PAD) for diffusion-based language models. PAD introduces Multi-Point Attention Attack, which guides parallel generative processes toward harmful outputs that inspired by affirmative response patterns in LLMs. Experimental evaluations across four LLDMs demonstrate that PAD achieves jailbreak attack success rates by 97%, revealing significant safety vulnerabilities. Furthermore, compared to autoregressive LLMs of the same size, LLDMs increase the harmful generation speed by 2x, significantly highlighting risks of uncontrolled misuse.Through comprehensive analysis, we provide an investigation into LLDM architecture, offering critical insights for the secure deployment of diffusion-based language models.



## **2. Blind Spot Navigation: Evolutionary Discovery of Sensitive Semantic Concepts for LVLMs**

cs.CV

The paper needs major revisions, so it is being withdrawn

**SubmitDate**: 2025-07-25    [abs](http://arxiv.org/abs/2505.15265v2) [paper-pdf](http://arxiv.org/pdf/2505.15265v2)

**Authors**: Zihao Pan, Yu Tong, Weibin Wu, Jingyi Wang, Lifeng Chen, Zhe Zhao, Jiajia Wei, Yitong Qiao, Zibin Zheng

**Abstract**: Adversarial attacks aim to generate malicious inputs that mislead deep models, but beyond causing model failure, they cannot provide certain interpretable information such as ``\textit{What content in inputs make models more likely to fail?}'' However, this information is crucial for researchers to specifically improve model robustness. Recent research suggests that models may be particularly sensitive to certain semantics in visual inputs (such as ``wet,'' ``foggy''), making them prone to errors. Inspired by this, in this paper we conducted the first exploration on large vision-language models (LVLMs) and found that LVLMs indeed are susceptible to hallucinations and various errors when facing specific semantic concepts in images. To efficiently search for these sensitive concepts, we integrated large language models (LLMs) and text-to-image (T2I) models to propose a novel semantic evolution framework. Randomly initialized semantic concepts undergo LLM-based crossover and mutation operations to form image descriptions, which are then converted by T2I models into visual inputs for LVLMs. The task-specific performance of LVLMs on each input is quantified as fitness scores for the involved semantics and serves as reward signals to further guide LLMs in exploring concepts that induce LVLMs. Extensive experiments on seven mainstream LVLMs and two multimodal tasks demonstrate the effectiveness of our method. Additionally, we provide interesting findings about the sensitive semantics of LVLMs, aiming to inspire further in-depth research.



## **3. KGV: Integrating Large Language Models with Knowledge Graphs for Cyber Threat Intelligence Credibility Assessment**

cs.CR

**SubmitDate**: 2025-07-25    [abs](http://arxiv.org/abs/2408.08088v2) [paper-pdf](http://arxiv.org/pdf/2408.08088v2)

**Authors**: Zongzong Wu, Fengxiao Tang, Ming Zhao, Yufeng Li

**Abstract**: Cyber threat intelligence (CTI) is a crucial tool to prevent sophisticated, organized, and weaponized cyber attacks. However, few studies have focused on the credibility assessment of CTI, and this work still requires manual analysis by cybersecurity experts. In this paper, we propose Knowledge Graph-based Verifier (KGV), the first framework integrating large language models (LLMs) with simple structured knowledge graphs (KGs) for automated CTI credibility assessment. Unlike entity-centric KGs, KGV constructs paragraph-level semantic graphs where nodes represent text segments connected through similarity analysis, which effectively enhances the semantic understanding ability of the model, reduces KG density and greatly improves response speed. Experimental results demonstrate that our KGV outperforms state-of-the-art fact reasoning methods on the CTI-200 dataset, achieving a 5.7\% improvement in F1. Additionally, it shows strong scalability on factual QA and fake news detection datasets. Compared to entity-based knowledge graphs (KGs) for equivalent-length texts, our structurally simple KG reduces node quantities by nearly two-thirds while boosting precision by 1.7\% and cutting response time by 46.7\%. In addition, we have created and publicly released the first CTI credibility assessment dataset, CTI-200. Distinct from CTI identification datasets, CTI-200 refines CTI summaries and key sentences to focus specifically on credibility assessment.



## **4. Kill two birds with one stone: generalized and robust AI-generated text detection via dynamic perturbations**

cs.CL

Accepted by NAACL 2025 main conference

**SubmitDate**: 2025-07-25    [abs](http://arxiv.org/abs/2504.21019v2) [paper-pdf](http://arxiv.org/pdf/2504.21019v2)

**Authors**: Yinghan Zhou, Juan Wen, Wanli Peng, Yiming Xue, Ziwei Zhang, Zhengxian Wu

**Abstract**: The growing popularity of large language models has raised concerns regarding the potential to misuse AI-generated text (AIGT). It becomes increasingly critical to establish an excellent AIGT detection method with high generalization and robustness. However, existing methods either focus on model generalization or concentrate on robustness. The unified mechanism, to simultaneously address the challenges of generalization and robustness, is less explored. In this paper, we argue that robustness can be view as a specific form of domain shift, and empirically reveal an intrinsic mechanism for model generalization of AIGT detection task. Then, we proposed a novel AIGT detection method (DP-Net) via dynamic perturbations introduced by a reinforcement learning with elaborated reward and action. Experimentally, extensive results show that the proposed DP-Net significantly outperforms some state-of-the-art AIGT detection methods for generalization capacity in three cross-domain scenarios. Meanwhile, the DP-Net achieves best robustness under two text adversarial attacks. The code is publicly available at https://github.com/CAU-ISS-Lab/AIGT-Detection-Evade-Detection/tree/main/DP-Net.



## **5. Model Tampering Attacks Enable More Rigorous Evaluations of LLM Capabilities**

cs.CR

Accepted to TMLR

**SubmitDate**: 2025-07-24    [abs](http://arxiv.org/abs/2502.05209v4) [paper-pdf](http://arxiv.org/pdf/2502.05209v4)

**Authors**: Zora Che, Stephen Casper, Robert Kirk, Anirudh Satheesh, Stewart Slocum, Lev E McKinney, Rohit Gandikota, Aidan Ewart, Domenic Rosati, Zichu Wu, Zikui Cai, Bilal Chughtai, Yarin Gal, Furong Huang, Dylan Hadfield-Menell

**Abstract**: Evaluations of large language model (LLM) risks and capabilities are increasingly being incorporated into AI risk management and governance frameworks. Currently, most risk evaluations are conducted by designing inputs that elicit harmful behaviors from the system. However, this approach suffers from two limitations. First, input-output evaluations cannot fully evaluate realistic risks from open-weight models. Second, the behaviors identified during any particular input-output evaluation can only lower-bound the model's worst-possible-case input-output behavior. As a complementary method for eliciting harmful behaviors, we propose evaluating LLMs with model tampering attacks which allow for modifications to latent activations or weights. We pit state-of-the-art techniques for removing harmful LLM capabilities against a suite of 5 input-space and 6 model tampering attacks. In addition to benchmarking these methods against each other, we show that (1) model resilience to capability elicitation attacks lies on a low-dimensional robustness subspace; (2) the success rate of model tampering attacks can empirically predict and offer conservative estimates for the success of held-out input-space attacks; and (3) state-of-the-art unlearning methods can easily be undone within 16 steps of fine-tuning. Together, these results highlight the difficulty of suppressing harmful LLM capabilities and show that model tampering attacks enable substantially more rigorous evaluations than input-space attacks alone.



## **6. Noise Contrastive Estimation-based Matching Framework for Low-Resource Security Attack Pattern Recognition**

cs.LG

accepted at EACL 2024, in ARR October 2023

**SubmitDate**: 2025-07-24    [abs](http://arxiv.org/abs/2401.10337v4) [paper-pdf](http://arxiv.org/pdf/2401.10337v4)

**Authors**: Tu Nguyen, Nedim Šrndić, Alexander Neth

**Abstract**: Tactics, Techniques and Procedures (TTPs) represent sophisticated attack patterns in the cybersecurity domain, described encyclopedically in textual knowledge bases. Identifying TTPs in cybersecurity writing, often called TTP mapping, is an important and challenging task. Conventional learning approaches often target the problem in the classical multi-class or multilabel classification setting. This setting hinders the learning ability of the model due to a large number of classes (i.e., TTPs), the inevitable skewness of the label distribution and the complex hierarchical structure of the label space. We formulate the problem in a different learning paradigm, where the assignment of a text to a TTP label is decided by the direct semantic similarity between the two, thus reducing the complexity of competing solely over the large labeling space. To that end, we propose a neural matching architecture with an effective sampling-based learn-to-compare mechanism, facilitating the learning process of the matching model despite constrained resources.



## **7. A comprehensive study of LLM-based argument classification: from LLAMA through GPT-4o to Deepseek-R1**

cs.CL

**SubmitDate**: 2025-07-24    [abs](http://arxiv.org/abs/2507.08621v2) [paper-pdf](http://arxiv.org/pdf/2507.08621v2)

**Authors**: Marcin Pietroń, Rafał Olszowski, Jakub Gomułka, Filip Gampel, Andrzej Tomski

**Abstract**: Argument mining (AM) is an interdisciplinary research field that integrates insights from logic, philosophy, linguistics, rhetoric, law, psychology, and computer science. It involves the automatic identification and extraction of argumentative components, such as premises and claims, and the detection of relationships between them, such as support, attack, or neutrality. Recently, the field has advanced significantly, especially with the advent of large language models (LLMs), which have enhanced the efficiency of analyzing and extracting argument semantics compared to traditional methods and other deep learning models. There are many benchmarks for testing and verifying the quality of LLM, but there is still a lack of research and results on the operation of these models in publicly available argument classification databases. This paper presents a study of a selection of LLM's, using diverse datasets such as Args.me and UKP. The models tested include versions of GPT, Llama, and DeepSeek, along with reasoning-enhanced variants incorporating the Chain-of-Thoughts algorithm. The results indicate that ChatGPT-4o outperforms the others in the argument classification benchmarks. In case of models incorporated with reasoning capabilities, the Deepseek-R1 shows its superiority. However, despite their superiority, GPT-4o and Deepseek-R1 still make errors. The most common errors are discussed for all models. To our knowledge, the presented work is the first broader analysis of the mentioned datasets using LLM and prompt algorithms. The work also shows some weaknesses of known prompt algorithms in argument analysis, while indicating directions for their improvement. The added value of the work is the in-depth analysis of the available argument datasets and the demonstration of their shortcomings.



## **8. BadReasoner: Planting Tunable Overthinking Backdoors into Large Reasoning Models for Fun or Profit**

cs.CL

**SubmitDate**: 2025-07-24    [abs](http://arxiv.org/abs/2507.18305v1) [paper-pdf](http://arxiv.org/pdf/2507.18305v1)

**Authors**: Biao Yi, Zekun Fei, Jianing Geng, Tong Li, Lihai Nie, Zheli Liu, Yiming Li

**Abstract**: Large reasoning models (LRMs) have emerged as a significant advancement in artificial intelligence, representing a specialized class of large language models (LLMs) designed to tackle complex reasoning tasks. The defining characteristic of LRMs lies in their extensive chain-of-thought (CoT) reasoning capabilities. In this paper, we identify a previously unexplored attack vector against LRMs, which we term "overthinking backdoors". We advance this concept by proposing a novel tunable backdoor, which moves beyond simple on/off attacks to one where an attacker can precisely control the extent of the model's reasoning verbosity. Our attack is implemented through a novel data poisoning methodology. It pairs a tunable trigger-where the number of repetitions signals the desired intensity-with a correspondingly verbose CoT response. These responses are programmatically generated by instructing a teacher LLM to inject a controlled number of redundant refinement steps into a correct reasoning process. The approach preserves output correctness, which ensures stealth and establishes the attack as a pure resource-consumption vector. Extensive empirical results on various LRMs demonstrate that our method can reliably trigger a controllable, multi-fold increase in the length of the reasoning process, without degrading the final answer's correctness. Our source code is available at https://github.com/FZaKK/BadReasoner.



## **9. Auto-SGCR: Automated Generation of Smart Grid Cyber Range Using IEC 61850 Standard Models**

cs.CR

12 pages

**SubmitDate**: 2025-07-24    [abs](http://arxiv.org/abs/2507.18249v1) [paper-pdf](http://arxiv.org/pdf/2507.18249v1)

**Authors**: Muhammad M. Roomi, S. M. Suhail Hussain, Ee-Chien Chang, David M. Nicol, Daisuke Mashima

**Abstract**: Digitalization of power grids have made them increasingly susceptible to cyber-attacks in the past decade. Iterative cybersecurity testing is indispensable to counter emerging attack vectors and to ensure dependability of critical infrastructure. Furthermore, these can be used to evaluate cybersecurity configuration, effectiveness of the cybersecurity measures against various attack vectors, as well as to train smart grid cybersecurity experts defending the system. Enabling extensive experiments narrows the gap between academic research and production environment. A high-fidelity cyber range is vital as it is often infeasible to conduct such experiments and training using production environment. However, the design and implementation of cyber range requires extensive domain knowledge of physical and cyber aspect of the infrastructure. Furthermore, costs incurred for setup and maintenance of cyber range are significant. Moreover, most existing smart grid cyber ranges are designed as a one-off, proprietary system, and are limited in terms of configurability, accessibility, portability, and reproducibility. To address these challenges, an automated Smart grid Cyber Range generation framework is presented in this paper. Initially a human-/machine-friendly, XML-based modeling language called Smart Grid Modeling Language was defined, which incorporates IEC 61850 System Configuration Language files. Subsequently, a toolchain to parse SG-ML model files and automatically instantiate a functional smart grid cyber range was developed. The developed SG-ML models can be easily shared and/or modified to reproduce or customize for any cyber range. The application of Auto-SGCR is demonstrated through case studies with large-scale substation models. The toolchain along with example SG-ML models have been open-sourced.



## **10. Safeguarding RAG Pipelines with GMTP: A Gradient-based Masked Token Probability Method for Poisoned Document Detection**

cs.CL

18 pages, accepted to ACL Findings 2025

**SubmitDate**: 2025-07-24    [abs](http://arxiv.org/abs/2507.18202v1) [paper-pdf](http://arxiv.org/pdf/2507.18202v1)

**Authors**: San Kim, Jonghwi Kim, Yejin Jeon, Gary Geunbae Lee

**Abstract**: Retrieval-Augmented Generation (RAG) enhances Large Language Models (LLMs) by providing external knowledge for accurate and up-to-date responses. However, this reliance on external sources exposes a security risk, attackers can inject poisoned documents into the knowledge base to steer the generation process toward harmful or misleading outputs. In this paper, we propose Gradient-based Masked Token Probability (GMTP), a novel defense method to detect and filter out adversarially crafted documents. Specifically, GMTP identifies high-impact tokens by examining gradients of the retriever's similarity function. These key tokens are then masked, and their probabilities are checked via a Masked Language Model (MLM). Since injected tokens typically exhibit markedly low masked-token probabilities, this enables GMTP to easily detect malicious documents and achieve high-precision filtering. Experiments demonstrate that GMTP is able to eliminate over 90% of poisoned content while retaining relevant documents, thus maintaining robust retrieval and generation performance across diverse datasets and adversarial settings.



## **11. Policy Disruption in Reinforcement Learning:Adversarial Attack with Large Language Models and Critical State Identification**

cs.LG

**SubmitDate**: 2025-07-24    [abs](http://arxiv.org/abs/2507.18113v1) [paper-pdf](http://arxiv.org/pdf/2507.18113v1)

**Authors**: Junyong Jiang, Buwei Tian, Chenxing Xu, Songze Li, Lu Dong

**Abstract**: Reinforcement learning (RL) has achieved remarkable success in fields like robotics and autonomous driving, but adversarial attacks designed to mislead RL systems remain challenging. Existing approaches often rely on modifying the environment or policy, limiting their practicality. This paper proposes an adversarial attack method in which existing agents in the environment guide the target policy to output suboptimal actions without altering the environment. We propose a reward iteration optimization framework that leverages large language models (LLMs) to generate adversarial rewards explicitly tailored to the vulnerabilities of the target agent, thereby enhancing the effectiveness of inducing the target agent toward suboptimal decision-making. Additionally, a critical state identification algorithm is designed to pinpoint the target agent's most vulnerable states, where suboptimal behavior from the victim leads to significant degradation in overall performance. Experimental results in diverse environments demonstrate the superiority of our method over existing approaches.



## **12. RECALLED: An Unbounded Resource Consumption Attack on Large Vision-Language Models**

cs.CR

**SubmitDate**: 2025-07-24    [abs](http://arxiv.org/abs/2507.18053v1) [paper-pdf](http://arxiv.org/pdf/2507.18053v1)

**Authors**: Haoran Gao, Yuanhe Zhang, Zhenhong Zhou, Lei Jiang, Fanyu Meng, Yujia Xiao, Kun Wang, Yang Liu, Junlan Feng

**Abstract**: Resource Consumption Attacks (RCAs) have emerged as a significant threat to the deployment of Large Language Models (LLMs). With the integration of vision modalities, additional attack vectors exacerbate the risk of RCAs in large vision-language models (LVLMs). However, existing red-teaming studies have largely overlooked visual inputs as a potential attack surface, resulting in insufficient mitigation strategies against RCAs in LVLMs. To address this gap, we propose RECALLED (\textbf{RE}source \textbf{C}onsumption \textbf{A}ttack on \textbf{L}arge Vision-\textbf{L}anguag\textbf{E} Mo\textbf{D}els), the first approach for exploiting visual modalities to trigger unbounded RCAs red-teaming. First, we present \textit{Vision Guided Optimization}, a fine-grained pixel-level optimization, to obtain \textit{Output Recall} adversarial perturbations, which can induce repeating output. Then, we inject the perturbations into visual inputs, triggering unbounded generations to achieve the goal of RCAs. Additionally, we introduce \textit{Multi-Objective Parallel Losses} to generate universal attack templates and resolve optimization conflicts when intending to implement parallel attacks. Empirical results demonstrate that RECALLED increases service response latency by over 26 $\uparrow$, resulting in an additional 20\% increase in GPU utilization and memory consumption. Our study exposes security vulnerabilities in LVLMs and establishes a red-teaming framework that can facilitate future defense development against RCAs.



## **13. ViGText: Deepfake Image Detection with Vision-Language Model Explanations and Graph Neural Networks**

cs.CV

**SubmitDate**: 2025-07-24    [abs](http://arxiv.org/abs/2507.18031v1) [paper-pdf](http://arxiv.org/pdf/2507.18031v1)

**Authors**: Ahmad ALBarqawi, Mahmoud Nazzal, Issa Khalil, Abdallah Khreishah, NhatHai Phan

**Abstract**: The rapid rise of deepfake technology, which produces realistic but fraudulent digital content, threatens the authenticity of media. Traditional deepfake detection approaches often struggle with sophisticated, customized deepfakes, especially in terms of generalization and robustness against malicious attacks. This paper introduces ViGText, a novel approach that integrates images with Vision Large Language Model (VLLM) Text explanations within a Graph-based framework to improve deepfake detection. The novelty of ViGText lies in its integration of detailed explanations with visual data, as it provides a more context-aware analysis than captions, which often lack specificity and fail to reveal subtle inconsistencies. ViGText systematically divides images into patches, constructs image and text graphs, and integrates them for analysis using Graph Neural Networks (GNNs) to identify deepfakes. Through the use of multi-level feature extraction across spatial and frequency domains, ViGText captures details that enhance its robustness and accuracy to detect sophisticated deepfakes. Extensive experiments demonstrate that ViGText significantly enhances generalization and achieves a notable performance boost when it detects user-customized deepfakes. Specifically, average F1 scores rise from 72.45% to 98.32% under generalization evaluation, and reflects the model's superior ability to generalize to unseen, fine-tuned variations of stable diffusion models. As for robustness, ViGText achieves an increase of 11.1% in recall compared to other deepfake detection approaches. When facing targeted attacks that exploit its graph-based architecture, ViGText limits classification performance degradation to less than 4%. ViGText uses detailed visual and textual analysis to set a new standard for detecting deepfakes, helping ensure media authenticity and information integrity.



## **14. Evaluating the Performance of AI Text Detectors, Few-Shot and Chain-of-Thought Prompting Using DeepSeek Generated Text**

cs.CL

**SubmitDate**: 2025-07-23    [abs](http://arxiv.org/abs/2507.17944v1) [paper-pdf](http://arxiv.org/pdf/2507.17944v1)

**Authors**: Hulayyil Alshammari, Praveen Rao

**Abstract**: Large language models (LLMs) have rapidly transformed the creation of written materials. LLMs have led to questions about writing integrity, thereby driving the creation of artificial intelligence (AI) detection technologies. Adversarial attacks, such as standard and humanized paraphrasing, inhibit detectors' ability to detect machine-generated text. Previous studies have mainly focused on ChatGPT and other well-known LLMs and have shown varying accuracy across detectors. However, there is a clear gap in the literature about DeepSeek, a recently published LLM. Therefore, in this work, we investigate whether six generally accessible AI detection tools -- AI Text Classifier, Content Detector AI, Copyleaks, QuillBot, GPT-2, and GPTZero -- can consistently recognize text generated by DeepSeek. The detectors were exposed to the aforementioned adversarial attacks. We also considered DeepSeek as a detector by performing few-shot prompting and chain-of-thought reasoning (CoT) for classifying AI and human-written text. We collected 49 human-authored question-answer pairs from before the LLM era and generated matching responses using DeepSeek-v3, producing 49 AI-generated samples. Then, we applied adversarial techniques such as paraphrasing and humanizing to add 196 more samples. These were used to challenge detector robustness and assess accuracy impact. While QuillBot and Copyleaks showed near-perfect performance on original and paraphrased DeepSeek text, others -- particularly AI Text Classifier and GPT-2 -- showed inconsistent results. The most effective attack was humanization, reducing accuracy to 71% for Copyleaks, 58% for QuillBot, and 52% for GPTZero. Few-shot and CoT prompting showed high accuracy, with the best five-shot result misclassifying only one of 49 samples (AI recall 96%, human recall 100%).



## **15. Weak-to-Strong Jailbreaking on Large Language Models**

cs.CL

ICML 2025

**SubmitDate**: 2025-07-23    [abs](http://arxiv.org/abs/2401.17256v5) [paper-pdf](http://arxiv.org/pdf/2401.17256v5)

**Authors**: Xuandong Zhao, Xianjun Yang, Tianyu Pang, Chao Du, Lei Li, Yu-Xiang Wang, William Yang Wang

**Abstract**: Large language models (LLMs) are vulnerable to jailbreak attacks - resulting in harmful, unethical, or biased text generations. However, existing jailbreaking methods are computationally costly. In this paper, we propose the weak-to-strong jailbreaking attack, an efficient inference time attack for aligned LLMs to produce harmful text. Our key intuition is based on the observation that jailbroken and aligned models only differ in their initial decoding distributions. The weak-to-strong attack's key technical insight is using two smaller models (a safe and an unsafe one) to adversarially modify a significantly larger safe model's decoding probabilities. We evaluate the weak-to-strong attack on 5 diverse open-source LLMs from 3 organizations. The results show our method can increase the misalignment rate to over 99% on two datasets with just one forward pass per example. Our study exposes an urgent safety issue that needs to be addressed when aligning LLMs. As an initial attempt, we propose a defense strategy to protect against such attacks, but creating more advanced defenses remains challenging. The code for replicating the method is available at https://github.com/XuandongZhao/weak-to-strong



## **16. LoX: Low-Rank Extrapolation Robustifies LLM Safety Against Fine-tuning**

cs.LG

**SubmitDate**: 2025-07-23    [abs](http://arxiv.org/abs/2506.15606v2) [paper-pdf](http://arxiv.org/pdf/2506.15606v2)

**Authors**: Gabriel J. Perin, Runjin Chen, Xuxi Chen, Nina S. T. Hirata, Zhangyang Wang, Junyuan Hong

**Abstract**: Large Language Models (LLMs) have become indispensable in real-world applications. However, their widespread adoption raises significant safety concerns, particularly in responding to socially harmful questions. Despite substantial efforts to improve model safety through alignment, aligned models can still have their safety protections undermined by subsequent fine-tuning - even when the additional training data appears benign. In this paper, we empirically demonstrate that this vulnerability stems from the sensitivity of safety-critical low-rank subspaces in LLM parameters to fine-tuning. Building on this insight, we propose a novel training-free method, termed Low-Rank Extrapolation (LoX), to enhance safety robustness by extrapolating the safety subspace of an aligned LLM. Our experimental results confirm the effectiveness of LoX, demonstrating significant improvements in robustness against both benign and malicious fine-tuning attacks while preserving the model's adaptability to new tasks. For instance, LoX leads to 11% to 54% absolute reductions in attack success rates (ASR) facing benign or malicious fine-tuning attacks. By investigating the ASR landscape of parameters, we attribute the success of LoX to that the extrapolation moves LLM parameters to a flatter zone, thereby less sensitive to perturbations. The code is available at github.com/VITA-Group/LoX.



## **17. MEF: A Capability-Aware Multi-Encryption Framework for Evaluating Vulnerabilities in Black-Box Large Language Models**

cs.CL

**SubmitDate**: 2025-07-23    [abs](http://arxiv.org/abs/2505.23404v4) [paper-pdf](http://arxiv.org/pdf/2505.23404v4)

**Authors**: Mingyu Yu, Wei Wang, Yanjie Wei, Sujuan Qin, Fei Gao, Wenmin Li

**Abstract**: Recent advancements in adversarial jailbreak attacks have exposed critical vulnerabilities in Large Language Models (LLMs), enabling the circumvention of alignment safeguards through increasingly sophisticated prompt manipulations. Based on our experiments, we found that the effectiveness of jailbreak strategies is influenced by the comprehension ability of the attacked LLM. Building on this insight, we propose a capability-aware Multi-Encryption Framework (MEF) for evaluating vulnerabilities in black-box LLMs. Specifically, MEF first categorizes the comprehension ability level of the LLM, then applies different strategies accordingly: For models with limited comprehension ability, MEF adopts the Fu+En1 strategy, which integrates layered semantic mutations with an encryption technique, more effectively contributing to evasion of the LLM's defenses at the input and inference stages. For models with strong comprehension ability, MEF uses a more complex Fu+En1+En2 strategy, in which additional dual-ended encryption techniques are applied to the LLM's responses, further contributing to evasion of the LLM's defenses at the output stage. Experimental results demonstrate the effectiveness of our approach, achieving attack success rates of 98.9% on GPT-4o (29 May 2025 release) and 99.8% on GPT-4.1 (8 July 2025 release). Our work contributes to a deeper understanding of the vulnerabilities in current LLM alignment mechanisms.



## **18. Tab-MIA: A Benchmark Dataset for Membership Inference Attacks on Tabular Data in LLMs**

cs.CR

**SubmitDate**: 2025-07-23    [abs](http://arxiv.org/abs/2507.17259v1) [paper-pdf](http://arxiv.org/pdf/2507.17259v1)

**Authors**: Eyal German, Sagiv Antebi, Daniel Samira, Asaf Shabtai, Yuval Elovici

**Abstract**: Large language models (LLMs) are increasingly trained on tabular data, which, unlike unstructured text, often contains personally identifiable information (PII) in a highly structured and explicit format. As a result, privacy risks arise, since sensitive records can be inadvertently retained by the model and exposed through data extraction or membership inference attacks (MIAs). While existing MIA methods primarily target textual content, their efficacy and threat implications may differ when applied to structured data, due to its limited content, diverse data types, unique value distributions, and column-level semantics. In this paper, we present Tab-MIA, a benchmark dataset for evaluating MIAs on tabular data in LLMs and demonstrate how it can be used. Tab-MIA comprises five data collections, each represented in six different encoding formats. Using our Tab-MIA benchmark, we conduct the first evaluation of state-of-the-art MIA methods on LLMs finetuned with tabular data across multiple encoding formats. In the evaluation, we analyze the memorization behavior of pretrained LLMs on structured data derived from Wikipedia tables. Our findings show that LLMs memorize tabular data in ways that vary across encoding formats, making them susceptible to extraction via MIAs. Even when fine-tuned for as few as three epochs, models exhibit high vulnerability, with AUROC scores approaching 90% in most cases. Tab-MIA enables systematic evaluation of these risks and provides a foundation for developing privacy-preserving methods for tabular data in LLMs.



## **19. LLM4MEA: Data-free Model Extraction Attacks on Sequential Recommenders via Large Language Models**

cs.IR

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2507.16969v1) [paper-pdf](http://arxiv.org/pdf/2507.16969v1)

**Authors**: Shilong Zhao, Fei Sun, Kaike Zhang, Shaoling Jing, Du Su, Zhichao Shi, Zhiyi Yin, Huawei Shen, Xueqi Cheng

**Abstract**: Recent studies have demonstrated the vulnerability of sequential recommender systems to Model Extraction Attacks (MEAs). MEAs collect responses from recommender systems to replicate their functionality, enabling unauthorized deployments and posing critical privacy and security risks. Black-box attacks in prior MEAs are ineffective at exposing recommender system vulnerabilities due to random sampling in data selection, which leads to misaligned synthetic and real-world distributions. To overcome this limitation, we propose LLM4MEA, a novel model extraction method that leverages Large Language Models (LLMs) as human-like rankers to generate data. It generates data through interactions between the LLM ranker and target recommender system. In each interaction, the LLM ranker analyzes historical interactions to understand user behavior, and selects items from recommendations with consistent preferences to extend the interaction history, which serves as training data for MEA. Extensive experiments demonstrate that LLM4MEA significantly outperforms existing approaches in data quality and attack performance, reducing the divergence between synthetic and real-world data by up to 64.98% and improving MEA performance by 44.82% on average. From a defensive perspective, we propose a simple yet effective defense strategy and identify key hyperparameters of recommender systems that can mitigate the risk of MEAs.



## **20. More is Less: The Pitfalls of Multi-Model Synthetic Preference Data in DPO Safety Alignment**

cs.AI

This version includes updated results and expanded discussion

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2504.02193v2) [paper-pdf](http://arxiv.org/pdf/2504.02193v2)

**Authors**: Yifan Wang, Runjin Chen, Bolian Li, David Cho, Yihe Deng, Ruqi Zhang, Tianlong Chen, Zhangyang Wang, Ananth Grama, Junyuan Hong

**Abstract**: Aligning large language models (LLMs) with human values is an increasingly critical step in post-training. Direct Preference Optimization (DPO) has emerged as a simple, yet effective alternative to reinforcement learning from human feedback (RLHF). Synthetic preference data with its low cost and high quality enable effective alignment through single- or multi-model generated preference data. Our study reveals a striking, safety-specific phenomenon associated with DPO alignment: Although multi-model generated data enhances performance on general tasks (ARC, Hellaswag, MMLU, TruthfulQA, Winogrande) by providing diverse responses, it also tends to facilitate reward hacking during training. This can lead to a high attack success rate (ASR) when models encounter jailbreaking prompts. The issue is particularly pronounced when employing stronger models like GPT-4o or larger models in the same family to generate chosen responses paired with target model self-generated rejected responses, resulting in dramatically poorer safety outcomes. Furthermore, with respect to safety, using solely self-generated responses (single-model generation) for both chosen and rejected pairs significantly outperforms configurations that incorporate responses from stronger models, whether used directly as chosen data or as part of a multi-model response pool. We demonstrate that multi-model preference data exhibits high linear separability between chosen and rejected responses, which allows models to exploit superficial cues rather than internalizing robust safety constraints. Our experiments, conducted on models from the Llama, Mistral, and Qwen families, consistently validate these findings.



## **21. When LLMs Copy to Think: Uncovering Copy-Guided Attacks in Reasoning LLMs**

cs.CR

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2507.16773v1) [paper-pdf](http://arxiv.org/pdf/2507.16773v1)

**Authors**: Yue Li, Xiao Li, Hao Wu, Yue Zhang, Fengyuan Xu, Xiuzhen Cheng, Sheng Zhong

**Abstract**: Large Language Models (LLMs) have become integral to automated code analysis, enabling tasks such as vulnerability detection and code comprehension. However, their integration introduces novel attack surfaces. In this paper, we identify and investigate a new class of prompt-based attacks, termed Copy-Guided Attacks (CGA), which exploit the inherent copying tendencies of reasoning-capable LLMs. By injecting carefully crafted triggers into external code snippets, adversaries can induce the model to replicate malicious content during inference. This behavior enables two classes of vulnerabilities: inference length manipulation, where the model generates abnormally short or excessively long reasoning traces; and inference result manipulation, where the model produces misleading or incorrect conclusions. We formalize CGA as an optimization problem and propose a gradient-based approach to synthesize effective triggers. Empirical evaluation on state-of-the-art reasoning LLMs shows that CGA reliably induces infinite loops, premature termination, false refusals, and semantic distortions in code analysis tasks. While highly effective in targeted settings, we observe challenges in generalizing CGA across diverse prompts due to computational constraints, posing an open question for future research. Our findings expose a critical yet underexplored vulnerability in LLM-powered development pipelines and call for urgent advances in prompt-level defense mechanisms.



## **22. From Text to Actionable Intelligence: Automating STIX Entity and Relationship Extraction**

cs.CR

This paper is accepted at RAID 2025

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2507.16576v1) [paper-pdf](http://arxiv.org/pdf/2507.16576v1)

**Authors**: Ahmed Lekssays, Husrev Taha Sencar, Ting Yu

**Abstract**: Sharing methods of attack and their effectiveness is a cornerstone of building robust defensive systems. Threat analysis reports, produced by various individuals and organizations, play a critical role in supporting security operations and combating emerging threats. To enhance the timeliness and automation of threat intelligence sharing, several standards have been established, with the Structured Threat Information Expression (STIX) framework emerging as one of the most widely adopted. However, generating STIX-compatible data from unstructured security text remains a largely manual, expert-driven process. To address this challenge, we introduce AZERG, a tool designed to assist security analysts in automatically generating structured STIX representations. To achieve this, we adapt general-purpose large language models for the specific task of extracting STIX-formatted threat data. To manage the complexity, the task is divided into four subtasks: entity detection (T1), entity type identification (T2), related pair detection (T3), and relationship type identification (T4). We apply task-specific fine-tuning to accurately extract relevant entities and infer their relationships in accordance with the STIX specification. To address the lack of training data, we compiled a comprehensive dataset with 4,011 entities and 2,075 relationships extracted from 141 full threat analysis reports, all annotated in alignment with the STIX standard. Our models achieved F1-scores of 84.43% for T1, 88.49% for T2, 95.47% for T3, and 84.60% for T4 in real-world scenarios. We validated their performance against a range of open- and closed-parameter models, as well as state-of-the-art methods, demonstrating improvements of 2-25% across tasks.



## **23. Depth Gives a False Sense of Privacy: LLM Internal States Inversion**

cs.CR

Accepted by USENIX Security 2025. Please cite this paper as "Tian  Dong, Yan Meng, Shaofeng Li, Guoxing Chen, Zhen Liu, Haojin Zhu. Depth Gives  a False Sense of Privacy: LLM Internal States Inversion. In the 34th USENIX  Security Symposium (USENIX Security '25)."

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2507.16372v1) [paper-pdf](http://arxiv.org/pdf/2507.16372v1)

**Authors**: Tian Dong, Yan Meng, Shaofeng Li, Guoxing Chen, Zhen Liu, Haojin Zhu

**Abstract**: Large Language Models (LLMs) are increasingly integrated into daily routines, yet they raise significant privacy and safety concerns. Recent research proposes collaborative inference, which outsources the early-layer inference to ensure data locality, and introduces model safety auditing based on inner neuron patterns. Both techniques expose the LLM's Internal States (ISs), which are traditionally considered irreversible to inputs due to optimization challenges and the highly abstract representations in deep layers. In this work, we challenge this assumption by proposing four inversion attacks that significantly improve the semantic similarity and token matching rate of inverted inputs. Specifically, we first develop two white-box optimization-based attacks tailored for low-depth and high-depth ISs. These attacks avoid local minima convergence, a limitation observed in prior work, through a two-phase inversion process. Then, we extend our optimization attack under more practical black-box weight access by leveraging the transferability between the source and the derived LLMs. Additionally, we introduce a generation-based attack that treats inversion as a translation task, employing an inversion model to reconstruct inputs. Extensive evaluation of short and long prompts from medical consulting and coding assistance datasets and 6 LLMs validates the effectiveness of our inversion attacks. Notably, a 4,112-token long medical consulting prompt can be nearly perfectly inverted with 86.88 F1 token matching from the middle layer of Llama-3 model. Finally, we evaluate four practical defenses that we found cannot perfectly prevent ISs inversion and draw conclusions for future mitigation design.



## **24. Can Indirect Prompt Injection Attacks Be Detected and Removed?**

cs.CR

To Appear in ACL 2025

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2502.16580v3) [paper-pdf](http://arxiv.org/pdf/2502.16580v3)

**Authors**: Yulin Chen, Haoran Li, Yuan Sui, Yufei He, Yue Liu, Yangqiu Song, Bryan Hooi

**Abstract**: Prompt injection attacks manipulate large language models (LLMs) by misleading them to deviate from the original input instructions and execute maliciously injected instructions, because of their instruction-following capabilities and inability to distinguish between the original input instructions and maliciously injected instructions. To defend against such attacks, recent studies have developed various detection mechanisms. If we restrict ourselves specifically to works which perform detection rather than direct defense, most of them focus on direct prompt injection attacks, while there are few works for the indirect scenario, where injected instructions are indirectly from external tools, such as a search engine. Moreover, current works mainly investigate injection detection methods and pay less attention to the post-processing method that aims to mitigate the injection after detection. In this paper, we investigate the feasibility of detecting and removing indirect prompt injection attacks, and we construct a benchmark dataset for evaluation. For detection, we assess the performance of existing LLMs and open-source detection models, and we further train detection models using our crafted training datasets. For removal, we evaluate two intuitive methods: (1) the segmentation removal method, which segments the injected document and removes parts containing injected instructions, and (2) the extraction removal method, which trains an extraction model to identify and remove injected instructions.



## **25. ShadowCode: Towards (Automatic) External Prompt Injection Attack against Code LLMs**

cs.CR

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2407.09164v6) [paper-pdf](http://arxiv.org/pdf/2407.09164v6)

**Authors**: Yuchen Yang, Yiming Li, Hongwei Yao, Bingrun Yang, Yiling He, Tianwei Zhang, Dacheng Tao, Zhan Qin

**Abstract**: Recent advancements have led to the widespread adoption of code-oriented large language models (Code LLMs) for programming tasks. Despite their success in deployment, their security research is left far behind. This paper introduces a new attack paradigm: (automatic) external prompt injection against Code LLMs, where attackers generate concise, non-functional induced perturbations and inject them within a victim's code context. These induced perturbations can be disseminated through commonly used dependencies (e.g., packages or RAG's knowledge base), manipulating Code LLMs to achieve malicious objectives during the code completion process. Compared to existing attacks, this method is more realistic and threatening: it does not necessitate control over the model's training process, unlike backdoor attacks, and can achieve specific malicious objectives that are challenging for adversarial attacks. Furthermore, we propose ShadowCode, a simple yet effective method that automatically generates induced perturbations based on code simulation to achieve effective and stealthy external prompt injection. ShadowCode designs its perturbation optimization objectives by simulating realistic code contexts and employs a greedy optimization approach with two enhancement modules: forward reasoning enhancement and keyword-based perturbation design. We evaluate our method across 13 distinct malicious objectives, generating 31 threat cases spanning three popular programming languages. Our results demonstrate that ShadowCode successfully attacks three representative open-source Code LLMs (achieving up to a 97.9% attack success rate) and two mainstream commercial Code LLM-integrated applications (with over 90% attack success rate) across all threat cases, using only a 12-token non-functional induced perturbation. The code is available at https://github.com/LianPing-cyber/ShadowCodeEPI.



## **26. Defense Against Prompt Injection Attack by Leveraging Attack Techniques**

cs.CR

To Appear in ACL 2025

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2411.00459v5) [paper-pdf](http://arxiv.org/pdf/2411.00459v5)

**Authors**: Yulin Chen, Haoran Li, Zihao Zheng, Yangqiu Song, Dekai Wu, Bryan Hooi

**Abstract**: With the advancement of technology, large language models (LLMs) have achieved remarkable performance across various natural language processing (NLP) tasks, powering LLM-integrated applications like Microsoft Copilot. However, as LLMs continue to evolve, new vulnerabilities, especially prompt injection attacks arise. These attacks trick LLMs into deviating from the original input instructions and executing the attacker's instructions injected in data content, such as retrieved results. Recent attack methods leverage LLMs' instruction-following abilities and their inabilities to distinguish instructions injected in the data content, and achieve a high attack success rate (ASR). When comparing the attack and defense methods, we interestingly find that they share similar design goals, of inducing the model to ignore unwanted instructions and instead to execute wanted instructions. Therefore, we raise an intuitive question: Could these attack techniques be utilized for defensive purposes? In this paper, we invert the intention of prompt injection methods to develop novel defense methods based on previous training-free attack methods, by repeating the attack process but with the original input instruction rather than the injected instruction. Our comprehensive experiments demonstrate that our defense techniques outperform existing training-free defense approaches, achieving state-of-the-art results.



## **27. CompLeak: Deep Learning Model Compression Exacerbates Privacy Leakage**

cs.CR

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2507.16872v1) [paper-pdf](http://arxiv.org/pdf/2507.16872v1)

**Authors**: Na Li, Yansong Gao, Hongsheng Hu, Boyu Kuang, Anmin Fu

**Abstract**: Model compression is crucial for minimizing memory storage and accelerating inference in deep learning (DL) models, including recent foundation models like large language models (LLMs). Users can access different compressed model versions according to their resources and budget. However, while existing compression operations primarily focus on optimizing the trade-off between resource efficiency and model performance, the privacy risks introduced by compression remain overlooked and insufficiently understood.   In this work, through the lens of membership inference attack (MIA), we propose CompLeak, the first privacy risk evaluation framework examining three widely used compression configurations that are pruning, quantization, and weight clustering supported by the commercial model compression framework of Google's TensorFlow-Lite (TF-Lite) and Facebook's PyTorch Mobile. CompLeak has three variants, given available access to the number of compressed models and original model. CompLeakNR starts by adopting existing MIA methods to attack a single compressed model, and identifies that different compressed models influence members and non-members differently. When the original model and one compressed model are available, CompLeakSR leverages the compressed model as a reference to the original model and uncovers more privacy by combining meta information (e.g., confidence vector) from both models. When multiple compressed models are available with/without accessing the original model, CompLeakMR innovatively exploits privacy leakage info from multiple compressed versions to substantially signify the overall privacy leakage. We conduct extensive experiments on seven diverse model architectures (from ResNet to foundation models of BERT and GPT-2), and six image and textual benchmark datasets.



## **28. OMNISEC: LLM-Driven Provenance-based Intrusion Detection via Retrieval-Augmented Behavior Prompting**

cs.CR

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2503.03108v4) [paper-pdf](http://arxiv.org/pdf/2503.03108v4)

**Authors**: Wenrui Cheng, Tiantian Zhu, Shunan Jing, Jian-Ping Mei, Mingjun Ma, Jiaobo Jin, Zhengqiu Weng

**Abstract**: Recently, Provenance-based Intrusion Detection Systems (PIDSes) have been widely used for endpoint threat analysis. These studies can be broadly categorized into rule-based detection systems and learning-based detection systems. Among these, due to the evolution of attack techniques, rules cannot dynamically model all the characteristics of attackers. As a result, such systems often face false negatives. Learning-based detection systems are further divided into supervised learning and anomaly detection. The scarcity of attack samples hinders the usability and effectiveness of supervised learning-based detection systems in practical applications. Anomaly-based detection systems face a massive false positive problem because they cannot distinguish between changes in normal behavior and real attack behavior. The alert results of detection systems are closely related to the manual labor costs of subsequent security analysts. To reduce manual analysis time, we propose OMNISEC, which applies large language models (LLMs) to anomaly-based intrusion detection systems via retrieval-augmented behavior prompting. OMNISEC can identify abnormal nodes and corresponding abnormal events by constructing suspicious nodes and rare paths. By combining two external knowledge bases, OMNISEC uses Retrieval Augmented Generation (RAG) to enable the LLM to determine whether abnormal behavior is a real attack. Finally, OMNISEC can reconstruct the attack graph and restore the complete attack behavior chain of the attacker's intrusion. Experimental results show that OMNISEC outperforms state-of-the-art methods on public benchmark datasets.



## **29. Talking Like a Phisher: LLM-Based Attacks on Voice Phishing Classifiers**

cs.CR

Accepted by EAI ICDF2C 2025

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2507.16291v1) [paper-pdf](http://arxiv.org/pdf/2507.16291v1)

**Authors**: Wenhao Li, Selvakumar Manickam, Yung-wey Chong, Shankar Karuppayah

**Abstract**: Voice phishing (vishing) remains a persistent threat in cybersecurity, exploiting human trust through persuasive speech. While machine learning (ML)-based classifiers have shown promise in detecting malicious call transcripts, they remain vulnerable to adversarial manipulations that preserve semantic content. In this study, we explore a novel attack vector where large language models (LLMs) are leveraged to generate adversarial vishing transcripts that evade detection while maintaining deceptive intent. We construct a systematic attack pipeline that employs prompt engineering and semantic obfuscation to transform real-world vishing scripts using four commercial LLMs. The generated transcripts are evaluated against multiple ML classifiers trained on a real-world Korean vishing dataset (KorCCViD) with statistical testing. Our experiments reveal that LLM-generated transcripts are both practically and statistically effective against ML-based classifiers. In particular, transcripts crafted by GPT-4o significantly reduce classifier accuracy (by up to 30.96%) while maintaining high semantic similarity, as measured by BERTScore. Moreover, these attacks are both time-efficient and cost-effective, with average generation times under 9 seconds and negligible financial cost per query. The results underscore the pressing need for more resilient vishing detection frameworks and highlight the imperative for LLM providers to enforce stronger safeguards against prompt misuse in adversarial social engineering contexts.



## **30. Quality Text, Robust Vision: The Role of Language in Enhancing Visual Robustness of Vision-Language Models**

cs.CV

ACMMM 2025 Accepted

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2507.16257v1) [paper-pdf](http://arxiv.org/pdf/2507.16257v1)

**Authors**: Futa Waseda, Saku Sugawara, Isao Echizen

**Abstract**: Defending pre-trained vision-language models (VLMs), such as CLIP, against adversarial attacks is crucial, as these models are widely used in diverse zero-shot tasks, including image classification. However, existing adversarial training (AT) methods for robust fine-tuning largely overlook the role of language in enhancing visual robustness. Specifically, (1) supervised AT methods rely on short texts (e.g., class labels) to generate adversarial perturbations, leading to overfitting to object classes in the training data, and (2) unsupervised AT avoids this overfitting but remains suboptimal against practical text-guided adversarial attacks due to its lack of semantic guidance. To address these limitations, we propose Quality Text-guided Adversarial Fine-Tuning (QT-AFT), which leverages high-quality captions during training to guide adversarial examples away from diverse semantics present in images. This enables the visual encoder to robustly recognize a broader range of image features even under adversarial noise, thereby enhancing robustness across diverse downstream tasks. QT-AFT overcomes the key weaknesses of prior methods -- overfitting in supervised AT and lack of semantic awareness in unsupervised AT -- achieving state-of-the-art zero-shot adversarial robustness and clean accuracy, evaluated across 16 zero-shot datasets. Furthermore, our comprehensive study uncovers several key insights into the role of language in enhancing vision robustness; for example, describing object properties in addition to object names further enhances zero-shot robustness. Our findings point to an urgent direction for future work -- centering high-quality linguistic supervision in robust visual representation learning.



## **31. BackdoorDM: A Comprehensive Benchmark for Backdoor Learning on Diffusion Model**

cs.CR

**SubmitDate**: 2025-07-21    [abs](http://arxiv.org/abs/2502.11798v2) [paper-pdf](http://arxiv.org/pdf/2502.11798v2)

**Authors**: Weilin Lin, Nanjun Zhou, Yanyun Wang, Jianze Li, Hui Xiong, Li Liu

**Abstract**: Backdoor learning is a critical research topic for understanding the vulnerabilities of deep neural networks. While the diffusion model (DM) has been broadly deployed in public over the past few years, the understanding of its backdoor vulnerability is still in its infancy compared to the extensive studies in discriminative models. Recently, many different backdoor attack and defense methods have been proposed for DMs, but a comprehensive benchmark for backdoor learning on DMs is still lacking. This absence makes it difficult to conduct fair comparisons and thorough evaluations of the existing approaches, thus hindering future research progress. To address this issue, we propose \textit{BackdoorDM}, the first comprehensive benchmark designed for backdoor learning on DMs. It comprises nine state-of-the-art (SOTA) attack methods, four SOTA defense strategies, and three useful visualization analysis tools. We first systematically classify and formulate the existing literature in a unified framework, focusing on three different backdoor attack types and five backdoor target types, which are restricted to a single type in discriminative models. Then, we systematically summarize the evaluation metrics for each type and propose a unified backdoor evaluation method based on multimodal large language model (MLLM). Finally, we conduct a comprehensive evaluation and highlight several important conclusions. We believe that BackdoorDM will help overcome current barriers and contribute to building a trustworthy artificial intelligence generated content (AIGC) community. The codes are released in https://github.com/linweiii/BackdoorDM.



## **32. Multi-Stage Prompt Inference Attacks on Enterprise LLM Systems**

cs.CR

26 pages

**SubmitDate**: 2025-07-21    [abs](http://arxiv.org/abs/2507.15613v1) [paper-pdf](http://arxiv.org/pdf/2507.15613v1)

**Authors**: Andrii Balashov, Olena Ponomarova, Xiaohua Zhai

**Abstract**: Large Language Models (LLMs) deployed in enterprise settings (e.g., as Microsoft 365 Copilot) face novel security challenges. One critical threat is prompt inference attacks: adversaries chain together seemingly benign prompts to gradually extract confidential data. In this paper, we present a comprehensive study of multi-stage prompt inference attacks in an enterprise LLM context. We simulate realistic attack scenarios where an attacker uses mild-mannered queries and indirect prompt injections to exploit an LLM integrated with private corporate data. We develop a formal threat model for these multi-turn inference attacks and analyze them using probability theory, optimization frameworks, and information-theoretic leakage bounds. The attacks are shown to reliably exfiltrate sensitive information from the LLM's context (e.g., internal SharePoint documents or emails), even when standard safety measures are in place.   We propose and evaluate defenses to counter such attacks, including statistical anomaly detection, fine-grained access control, prompt sanitization techniques, and architectural modifications to LLM deployment. Each defense is supported by mathematical analysis or experimental simulation. For example, we derive bounds on information leakage under differential privacy-based training and demonstrate an anomaly detection method that flags multi-turn attacks with high AUC. We also introduce an approach called "spotlighting" that uses input transformations to isolate untrusted prompt content, reducing attack success by an order of magnitude. Finally, we provide a formal proof of concept and empirical validation for a combined defense-in-depth strategy. Our work highlights that securing LLMs in enterprise settings requires moving beyond single-turn prompt filtering toward a holistic, multi-stage perspective on both attacks and defenses.



## **33. PiMRef: Detecting and Explaining Ever-evolving Spear Phishing Emails with Knowledge Base Invariants**

cs.CR

**SubmitDate**: 2025-07-21    [abs](http://arxiv.org/abs/2507.15393v1) [paper-pdf](http://arxiv.org/pdf/2507.15393v1)

**Authors**: Ruofan Liu, Yun Lin, Silas Yeo Shuen Yu, Xiwen Teoh, Zhenkai Liang, Jin Song Dong

**Abstract**: Phishing emails are a critical component of the cybercrime kill chain due to their wide reach and low cost. Their ever-evolving nature renders traditional rule-based and feature-engineered detectors ineffective in the ongoing arms race between attackers and defenders. The rise of large language models (LLMs) further exacerbates the threat, enabling attackers to craft highly convincing phishing emails at minimal cost.   This work demonstrates that LLMs can generate psychologically persuasive phishing emails tailored to victim profiles, successfully bypassing nearly all commercial and academic detectors. To defend against such threats, we propose PiMRef, the first reference-based phishing email detector that leverages knowledge-based invariants. Our core insight is that persuasive phishing emails often contain disprovable identity claims, which contradict real-world facts. PiMRef reframes phishing detection as an identity fact-checking task. Given an email, PiMRef (i) extracts the sender's claimed identity, (ii) verifies the legitimacy of the sender's domain against a predefined knowledge base, and (iii) detects call-to-action prompts that push user engagement. Contradictory claims are flagged as phishing indicators and serve as human-understandable explanations.   Compared to existing methods such as D-Fence, HelpHed, and ChatSpamDetector, PiMRef boosts precision by 8.8% with no loss in recall on standard benchmarks like Nazario and PhishPot. In a real-world evaluation of 10,183 emails across five university accounts over three years, PiMRef achieved 92.1% precision, 87.9% recall, and a median runtime of 0.05s, outperforming the state-of-the-art in both effectiveness and efficiency.



## **34. Transfer Attack for Bad and Good: Explain and Boost Adversarial Transferability across Multimodal Large Language Models**

cs.CV

This paper is accepted by ACM MM 2025

**SubmitDate**: 2025-07-21    [abs](http://arxiv.org/abs/2405.20090v5) [paper-pdf](http://arxiv.org/pdf/2405.20090v5)

**Authors**: Hao Cheng, Erjia Xiao, Jiayan Yang, Jinhao Duan, Yichi Wang, Jiahang Cao, Qiang Zhang, Le Yang, Kaidi Xu, Jindong Gu, Renjing Xu

**Abstract**: Multimodal Large Language Models (MLLMs) demonstrate exceptional performance in cross-modality interaction, yet they also suffer adversarial vulnerabilities. In particular, the transferability of adversarial examples remains an ongoing challenge. In this paper, we specifically analyze the manifestation of adversarial transferability among MLLMs and identify the key factors that influence this characteristic. We discover that the transferability of MLLMs exists in cross-LLM scenarios with the same vision encoder and indicate \underline{\textit{two key Factors}} that may influence transferability. We provide two semantic-level data augmentation methods, Adding Image Patch (AIP) and Typography Augment Transferability Method (TATM), which boost the transferability of adversarial examples across MLLMs. To explore the potential impact in the real world, we utilize two tasks that can have both negative and positive societal impacts: \ding{182} Harmful Content Insertion and \ding{183} Information Protection.



## **35. Scaling Decentralized Learning with FLock**

cs.LG

**SubmitDate**: 2025-07-21    [abs](http://arxiv.org/abs/2507.15349v1) [paper-pdf](http://arxiv.org/pdf/2507.15349v1)

**Authors**: Zehua Cheng, Rui Sun, Jiahao Sun, Yike Guo

**Abstract**: Fine-tuning the large language models (LLMs) are prevented by the deficiency of centralized control and the massive computing and communication overhead on the decentralized schemes. While the typical standard federated learning (FL) supports data privacy, the central server requirement creates a single point of attack and vulnerability to poisoning attacks. Generalizing the result in this direction to 70B-parameter models in the heterogeneous, trustless environments has turned out to be a huge, yet unbroken bottleneck. This paper introduces FLock, a decentralized framework for secure and efficient collaborative LLM fine-tuning. Integrating a blockchain-based trust layer with economic incentives, FLock replaces the central aggregator with a secure, auditable protocol for cooperation among untrusted parties. We present the first empirical validation of fine-tuning a 70B LLM in a secure, multi-domain, decentralized setting. Our experiments show the FLock framework defends against backdoor poisoning attacks that compromise standard FL optimizers and fosters synergistic knowledge transfer. The resulting models show a >68% reduction in adversarial attack success rates. The global model also demonstrates superior cross-domain generalization, outperforming models trained in isolation on their own specialized data.



## **36. From LLMs to MLLMs to Agents: A Survey of Emerging Paradigms in Jailbreak Attacks and Defenses within LLM Ecosystem**

cs.CR

**SubmitDate**: 2025-07-20    [abs](http://arxiv.org/abs/2506.15170v2) [paper-pdf](http://arxiv.org/pdf/2506.15170v2)

**Authors**: Yanxu Mao, Tiehan Cui, Peipei Liu, Datao You, Hongsong Zhu

**Abstract**: Large language models (LLMs) are rapidly evolving from single-modal systems to multimodal LLMs and intelligent agents, significantly expanding their capabilities while introducing increasingly severe security risks. This paper presents a systematic survey of the growing complexity of jailbreak attacks and corresponding defense mechanisms within the expanding LLM ecosystem. We first trace the developmental trajectory from LLMs to MLLMs and Agents, highlighting the core security challenges emerging at each stage. Next, we categorize mainstream jailbreak techniques from both the attack impact and visibility perspectives, and provide a comprehensive analysis of representative attack methods, related datasets, and evaluation metrics. On the defense side, we organize existing strategies based on response timing and technical approach, offering a structured understanding of their applicability and implementation. Furthermore, we identify key limitations in existing surveys, such as insufficient attention to agent-specific security issues, the absence of a clear taxonomy for hybrid jailbreak methods, a lack of detailed analysis of experimental setups, and outdated coverage of recent advancements. To address these limitations, we provide an updated synthesis of recent work and outline future research directions in areas such as dataset construction, evaluation framework optimization, and strategy generalization. Our study seeks to enhance the understanding of jailbreak mechanisms and facilitate the advancement of more resilient and adaptive defense strategies in the context of ever more capable LLMs.



## **37. Byzantine-Robust Decentralized Coordination of LLM Agents**

cs.DC

**SubmitDate**: 2025-07-20    [abs](http://arxiv.org/abs/2507.14928v1) [paper-pdf](http://arxiv.org/pdf/2507.14928v1)

**Authors**: Yongrae Jo, Chanik Park

**Abstract**: Collaboration among multiple large language model (LLM) agents is a promising approach to overcome inherent limitations of single-agent systems, such as hallucinations and single points of failure. As LLM agents are increasingly deployed on open blockchain platforms, multi-agent systems capable of tolerating malicious (Byzantine) agents have become essential.   Recent Byzantine-robust multi-agent systems typically rely on leader-driven coordination, which suffers from two major drawbacks. First, they are inherently vulnerable to targeted attacks against the leader. If consecutive leaders behave maliciously, the system repeatedly fails to achieve consensus, forcing new consensus rounds, which is particularly costly given the high latency of LLM invocations. Second, an underperforming proposal from the leader can be accepted as the final answer even when higher-quality alternatives are available, as existing methods finalize the leader's proposal once it receives a quorum of votes.   To address these issues, we propose DecentLLMs, a novel decentralized consensus approach for multi-agent LLM systems, where worker agents generate answers concurrently and evaluator agents independently score and rank these answers to select the best available one. This decentralized architecture enables faster consensus despite the presence of Byzantine agents and consistently selects higher-quality answers through Byzantine-robust aggregation techniques.   Experimental results demonstrate that DecentLLMs effectively tolerates Byzantine agents and significantly improves the quality of selected answers.



## **38. Configurable multi-agent framework for scalable and realistic testing of llm-based agents**

cs.AI

**SubmitDate**: 2025-07-19    [abs](http://arxiv.org/abs/2507.14705v1) [paper-pdf](http://arxiv.org/pdf/2507.14705v1)

**Authors**: Sai Wang, Senthilnathan Subramanian, Mudit Sahni, Praneeth Gone, Lingjie Meng, Xiaochen Wang, Nicolas Ferradas Bertoli, Tingxian Cheng, Jun Xu

**Abstract**: Large-language-model (LLM) agents exhibit complex, context-sensitive behaviour that quickly renders static benchmarks and ad-hoc manual testing obsolete.   We present Neo, a configurable, multi-agent framework that automates realistic, multi-turn evaluation of LLM-based systems. Neo couples a Question Generation Agent and an Evaluation Agent through a shared context-hub, allowing domain prompts, scenario controls and dynamic feedback to be composed modularly. Test inputs are sampled from a probabilistic state model spanning dialogue flow, user intent and emotional tone, enabling diverse, human-like conversations that adapt after every turn.   Applied to a production-grade Seller Financial Assistant chatbot, Neo (i) uncovered edge-case failures across five attack categories with a 3.3% break rate close to the 5.8% achieved by expert human red-teamers, and (ii) delivered 10-12X higher throughput, generating 180 coherent test questions in around 45 mins versus 16h of human effort. Beyond security probing, Neo's stochastic policies balanced topic coverage and conversational depth, yielding broader behavioural exploration than manually crafted scripts.   Neo therefore lays a foundation for scalable, self-evolving LLM QA: its agent interfaces, state controller and feedback loops are model-agnostic and extensible to richer factual-grounding and policy-compliance checks. We release the framework to facilitate reproducible, high-fidelity testing of emerging agentic systems.



## **39. Blackbox Dataset Inference for LLM**

cs.CR

**SubmitDate**: 2025-07-18    [abs](http://arxiv.org/abs/2507.03619v2) [paper-pdf](http://arxiv.org/pdf/2507.03619v2)

**Authors**: Ruikai Zhou, Kang Yang, Xun Chen, Wendy Hui Wang, Guanhong Tao, Jun Xu

**Abstract**: Today, the training of large language models (LLMs) can involve personally identifiable information and copyrighted material, incurring dataset misuse. To mitigate the problem of dataset misuse, this paper explores \textit{dataset inference}, which aims to detect if a suspect model $\mathcal{M}$ used a victim dataset $\mathcal{D}$ in training. Previous research tackles dataset inference by aggregating results of membership inference attacks (MIAs) -- methods to determine whether individual samples are a part of the training dataset. However, restricted by the low accuracy of MIAs, previous research mandates grey-box access to $\mathcal{M}$ to get intermediate outputs (probabilities, loss, perplexity, etc.) for obtaining satisfactory results. This leads to reduced practicality, as LLMs, especially those deployed for profits, have limited incentives to return the intermediate outputs.   In this paper, we propose a new method of dataset inference with only black-box access to the target model (i.e., assuming only the text-based responses of the target model are available). Our method is enabled by two sets of locally built reference models, one set involving $\mathcal{D}$ in training and the other not. By measuring which set of reference model $\mathcal{M}$ is closer to, we determine if $\mathcal{M}$ used $\mathcal{D}$ for training. Evaluations of real-world LLMs in the wild show that our method offers high accuracy in all settings and presents robustness against bypassing attempts.



## **40. From Words to Collisions: LLM-Guided Evaluation and Adversarial Generation of Safety-Critical Driving Scenarios**

cs.AI

Final Version and Paper Accepted at IEEE ITSC 2025

**SubmitDate**: 2025-07-18    [abs](http://arxiv.org/abs/2502.02145v4) [paper-pdf](http://arxiv.org/pdf/2502.02145v4)

**Authors**: Yuan Gao, Mattia Piccinini, Korbinian Moller, Amr Alanwar, Johannes Betz

**Abstract**: Ensuring the safety of autonomous vehicles requires virtual scenario-based testing, which depends on the robust evaluation and generation of safety-critical scenarios. So far, researchers have used scenario-based testing frameworks that rely heavily on handcrafted scenarios as safety metrics. To reduce the effort of human interpretation and overcome the limited scalability of these approaches, we combine Large Language Models (LLMs) with structured scenario parsing and prompt engineering to automatically evaluate and generate safety-critical driving scenarios. We introduce Cartesian and Ego-centric prompt strategies for scenario evaluation, and an adversarial generation module that modifies trajectories of risk-inducing vehicles (ego-attackers) to create critical scenarios. We validate our approach using a 2D simulation framework and multiple pre-trained LLMs. The results show that the evaluation module effectively detects collision scenarios and infers scenario safety. Meanwhile, the new generation module identifies high-risk agents and synthesizes realistic, safety-critical scenarios. We conclude that an LLM equipped with domain-informed prompting techniques can effectively evaluate and generate safety-critical driving scenarios, reducing dependence on handcrafted metrics. We release our open-source code and scenarios at: https://github.com/TUM-AVS/From-Words-to-Collisions.



## **41. TopicAttack: An Indirect Prompt Injection Attack via Topic Transition**

cs.CR

19 pages

**SubmitDate**: 2025-07-18    [abs](http://arxiv.org/abs/2507.13686v1) [paper-pdf](http://arxiv.org/pdf/2507.13686v1)

**Authors**: Yulin Chen, Haoran Li, Yuexin Li, Yue Liu, Yangqiu Song, Bryan Hooi

**Abstract**: Large language models (LLMs) have shown remarkable performance across a range of NLP tasks. However, their strong instruction-following capabilities and inability to distinguish instructions from data content make them vulnerable to indirect prompt injection attacks. In such attacks, instructions with malicious purposes are injected into external data sources, such as web documents. When LLMs retrieve this injected data through tools, such as a search engine and execute the injected instructions, they provide misled responses. Recent attack methods have demonstrated potential, but their abrupt instruction injection often undermines their effectiveness. Motivated by the limitations of existing attack methods, we propose TopicAttack, which prompts the LLM to generate a fabricated conversational transition prompt that gradually shifts the topic toward the injected instruction, making the injection smoother and enhancing the plausibility and success of the attack. Through comprehensive experiments, TopicAttack achieves state-of-the-art performance, with an attack success rate (ASR) over 90\% in most cases, even when various defense methods are applied. We further analyze its effectiveness by examining attention scores. We find that a higher injected-to-original attention ratio leads to a greater success probability, and our method achieves a much higher ratio than the baseline methods.



## **42. Invisible Textual Backdoor Attacks based on Dual-Trigger**

cs.CR

**SubmitDate**: 2025-07-18    [abs](http://arxiv.org/abs/2412.17531v3) [paper-pdf](http://arxiv.org/pdf/2412.17531v3)

**Authors**: Yang Hou, Qiuling Yue, Lujia Chai, Guozhao Liao, Wenbao Han, Wei Ou

**Abstract**: Backdoor attacks pose an important security threat to textual large language models. Exploring textual backdoor attacks not only helps reveal the potential security risks of models, but also promotes innovation and development of defense mechanisms. Currently, most textual backdoor attack methods are based on a single trigger. For example, inserting specific content into text as a trigger or changing the abstract text features to be a trigger. However, the adoption of this single-trigger mode makes the existing backdoor attacks subject to certain limitations: either they are easily identified by the existing defense strategies, or they have certain shortcomings in attack performance and in the construction of poisoned datasets. In order to solve these issues, a dual-trigger backdoor attack method is proposed in this paper. Specifically, we use two different attributes, syntax and mood (we use subjunctive mood as an example in this article), as two different triggers. It makes our backdoor attack method similar to a double landmine which can have completely different trigger conditions simultaneously. Therefore, this method not only improves the flexibility of trigger mode, but also enhances the robustness against defense detection. A large number of experimental results show that this method significantly outperforms the previous methods based on abstract features in attack performance, and achieves comparable attack performance (almost 100\% attack success rate) with the insertion-based method. In addition, in order to further improve the attack performance, we also give the construction method of the poisoned dataset.The code and data of this paper can be obtained at https://github.com/HoyaAm/Double-Landmines.



## **43. Paper Summary Attack: Jailbreaking LLMs through LLM Safety Papers**

cs.CL

**SubmitDate**: 2025-07-17    [abs](http://arxiv.org/abs/2507.13474v1) [paper-pdf](http://arxiv.org/pdf/2507.13474v1)

**Authors**: Liang Lin, Zhihao Xu, Xuehai Tang, Shi Liu, Biyu Zhou, Fuqing Zhu, Jizhong Han, Songlin Hu

**Abstract**: The safety of large language models (LLMs) has garnered significant research attention. In this paper, we argue that previous empirical studies demonstrate LLMs exhibit a propensity to trust information from authoritative sources, such as academic papers, implying new possible vulnerabilities. To verify this possibility, a preliminary analysis is designed to illustrate our two findings. Based on this insight, a novel jailbreaking method, Paper Summary Attack (\llmname{PSA}), is proposed. It systematically synthesizes content from either attack-focused or defense-focused LLM safety paper to construct an adversarial prompt template, while strategically infilling harmful query as adversarial payloads within predefined subsections. Extensive experiments show significant vulnerabilities not only in base LLMs, but also in state-of-the-art reasoning model like Deepseek-R1. PSA achieves a 97\% attack success rate (ASR) on well-aligned models like Claude3.5-Sonnet and an even higher 98\% ASR on Deepseek-R1. More intriguingly, our work has further revealed diametrically opposed vulnerability bias across different base models, and even between different versions of the same model, when exposed to either attack-focused or defense-focused papers. This phenomenon potentially indicates future research clues for both adversarial methodologies and safety alignment.Code is available at https://github.com/233liang/Paper-Summary-Attack



## **44. Automating Steering for Safe Multimodal Large Language Models**

cs.CL

Working in progress. 22 pages (8+ for main); 25 figures; 1 table

**SubmitDate**: 2025-07-17    [abs](http://arxiv.org/abs/2507.13255v1) [paper-pdf](http://arxiv.org/pdf/2507.13255v1)

**Authors**: Lyucheng Wu, Mengru Wang, Ziwen Xu, Tri Cao, Nay Oo, Bryan Hooi, Shumin Deng

**Abstract**: Recent progress in Multimodal Large Language Models (MLLMs) has unlocked powerful cross-modal reasoning abilities, but also raised new safety concerns, particularly when faced with adversarial multimodal inputs. To improve the safety of MLLMs during inference, we introduce a modular and adaptive inference-time intervention technology, AutoSteer, without requiring any fine-tuning of the underlying model. AutoSteer incorporates three core components: (1) a novel Safety Awareness Score (SAS) that automatically identifies the most safety-relevant distinctions among the model's internal layers; (2) an adaptive safety prober trained to estimate the likelihood of toxic outputs from intermediate representations; and (3) a lightweight Refusal Head that selectively intervenes to modulate generation when safety risks are detected. Experiments on LLaVA-OV and Chameleon across diverse safety-critical benchmarks demonstrate that AutoSteer significantly reduces the Attack Success Rate (ASR) for textual, visual, and cross-modal threats, while maintaining general abilities. These findings position AutoSteer as a practical, interpretable, and effective framework for safer deployment of multimodal AI systems.



## **45. Detecting LLM-generated Code with Subtle Modification by Adversarial Training**

cs.SE

**SubmitDate**: 2025-07-17    [abs](http://arxiv.org/abs/2507.13123v1) [paper-pdf](http://arxiv.org/pdf/2507.13123v1)

**Authors**: Xin Yin, Xinrui Li, Chao Ni, Xiaodan Xu, Xiaohu Yang

**Abstract**: With the rapid development of Large Language Models (LLMs), their powerful code-generation capabilities have been widely applied in tasks like code completion and automated development, demonstrating the value of improving coding efficiency. However, the extensive use of LLM-generated code also raises several new challenges. On the one hand, issues such as the regulation of code provenance, copyright disputes, and code quality have become increasingly concerning. How to effectively detect LLM-generated code and ensure its compliant and responsible use has become a critical and urgent issue. On the other hand, in practical applications, LLM-generated code is often subject to manual modifications, such as variable renaming or structural adjustments. Although some recent studies have proposed training-based and zero-shot methods for detecting LLM-generated code, these approaches show insufficient robustness when facing modified LLM-generated code, and there is a lack of an effective solution. To address the real-world scenario where LLM-generated code may undergo minor modifications, we propose CodeGPTSensor+, an enhanced version of CodeGPTSensor, which employs adversarial training to improve robustness against input perturbations. CodeGPTSensor+ integrates an adversarial sample generation module, Multi-objective Identifier and Structure Transformation (MIST), which systematically generates both high-quality and representative adversarial samples. This module effectively enhances the model's resistance against diverse adversarial attacks. Experimental results on the HMCorp dataset demonstrate that CodeGPTSensor+ significantly improves detection accuracy on the adversarial test set while maintaining high accuracy on the original test set, showcasing superior robustness compared to CodeGPTSensor.



## **46. MAD-Spear: A Conformity-Driven Prompt Injection Attack on Multi-Agent Debate Systems**

cs.CR

**SubmitDate**: 2025-07-17    [abs](http://arxiv.org/abs/2507.13038v1) [paper-pdf](http://arxiv.org/pdf/2507.13038v1)

**Authors**: Yu Cui, Hongyang Du

**Abstract**: Multi-agent debate (MAD) systems leverage collaborative interactions among large language models (LLMs) agents to improve reasoning capabilities. While recent studies have focused on increasing the accuracy and scalability of MAD systems, their security vulnerabilities have received limited attention. In this work, we introduce MAD-Spear, a targeted prompt injection attack that compromises a small subset of agents but significantly disrupts the overall MAD process. Manipulated agents produce multiple plausible yet incorrect responses, exploiting LLMs' conformity tendencies to propagate misinformation and degrade consensus quality. Furthermore, the attack can be composed with other strategies, such as communication attacks, to further amplify its impact by increasing the exposure of agents to incorrect responses. To assess MAD's resilience under attack, we propose a formal definition of MAD fault-tolerance and develop a comprehensive evaluation framework that jointly considers accuracy, consensus efficiency, and scalability. Extensive experiments on five benchmark datasets with varying difficulty levels demonstrate that MAD-Spear consistently outperforms the baseline attack in degrading system performance. Additionally, we observe that agent diversity substantially improves MAD performance in mathematical reasoning tasks, which challenges prior work suggesting that agent diversity has minimal impact on performance. These findings highlight the urgent need to improve the security in MAD design.



## **47. JailDAM: Jailbreak Detection with Adaptive Memory for Vision-Language Model**

cs.CR

**SubmitDate**: 2025-07-16    [abs](http://arxiv.org/abs/2504.03770v3) [paper-pdf](http://arxiv.org/pdf/2504.03770v3)

**Authors**: Yi Nian, Shenzhe Zhu, Yuehan Qin, Li Li, Ziyi Wang, Chaowei Xiao, Yue Zhao

**Abstract**: Multimodal large language models (MLLMs) excel in vision-language tasks but also pose significant risks of generating harmful content, particularly through jailbreak attacks. Jailbreak attacks refer to intentional manipulations that bypass safety mechanisms in models, leading to the generation of inappropriate or unsafe content. Detecting such attacks is critical to ensuring the responsible deployment of MLLMs. Existing jailbreak detection methods face three primary challenges: (1) Many rely on model hidden states or gradients, limiting their applicability to white-box models, where the internal workings of the model are accessible; (2) They involve high computational overhead from uncertainty-based analysis, which limits real-time detection, and (3) They require fully labeled harmful datasets, which are often scarce in real-world settings. To address these issues, we introduce a test-time adaptive framework called JAILDAM. Our method leverages a memory-based approach guided by policy-driven unsafe knowledge representations, eliminating the need for explicit exposure to harmful data. By dynamically updating unsafe knowledge during test-time, our framework improves generalization to unseen jailbreak strategies while maintaining efficiency. Experiments on multiple VLM jailbreak benchmarks demonstrate that JAILDAM delivers state-of-the-art performance in harmful content detection, improving both accuracy and speed.



## **48. SoK: Semantic Privacy in Large Language Models**

cs.CR

**SubmitDate**: 2025-07-16    [abs](http://arxiv.org/abs/2506.23603v2) [paper-pdf](http://arxiv.org/pdf/2506.23603v2)

**Authors**: Baihe Ma, Yanna Jiang, Xu Wang, Guangsheng Yu, Qin Wang, Caijun Sun, Chen Li, Xuelei Qi, Ying He, Wei Ni, Ren Ping Liu

**Abstract**: As Large Language Models (LLMs) are increasingly deployed in sensitive domains, traditional data privacy measures prove inadequate for protecting information that is implicit, contextual, or inferable - what we define as semantic privacy. This Systematization of Knowledge (SoK) introduces a lifecycle-centric framework to analyze how semantic privacy risks emerge across input processing, pretraining, fine-tuning, and alignment stages of LLMs. We categorize key attack vectors and assess how current defenses, such as differential privacy, embedding encryption, edge computing, and unlearning, address these threats. Our analysis reveals critical gaps in semantic-level protection, especially against contextual inference and latent representation leakage. We conclude by outlining open challenges, including quantifying semantic leakage, protecting multimodal inputs, balancing de-identification with generation quality, and ensuring transparency in privacy enforcement. This work aims to inform future research on designing robust, semantically aware privacy-preserving techniques for LLMs.



## **49. Thought Purity: Defense Paradigm For Chain-of-Thought Attack**

cs.LG

**SubmitDate**: 2025-07-16    [abs](http://arxiv.org/abs/2507.12314v1) [paper-pdf](http://arxiv.org/pdf/2507.12314v1)

**Authors**: Zihao Xue, Zhen Bi, Long Ma, Zhenlin Hu, Yan Wang, Zhenfang Liu, Qing Sheng, Jie Xiao, Jungang Lou

**Abstract**: While reinforcement learning-trained Large Reasoning Models (LRMs, e.g., Deepseek-R1) demonstrate advanced reasoning capabilities in the evolving Large Language Models (LLMs) domain, their susceptibility to security threats remains a critical vulnerability. This weakness is particularly evident in Chain-of-Thought (CoT) generation processes, where adversarial methods like backdoor prompt attacks can systematically subvert the model's core reasoning mechanisms. The emerging Chain-of-Thought Attack (CoTA) reveals this vulnerability through exploiting prompt controllability, simultaneously degrading both CoT safety and task performance with low-cost interventions. To address this compounded security-performance vulnerability, we propose Thought Purity (TP): a defense paradigm that systematically strengthens resistance to malicious content while preserving operational efficacy. Our solution achieves this through three synergistic components: (1) a safety-optimized data processing pipeline (2) reinforcement learning-enhanced rule constraints (3) adaptive monitoring metrics. Our approach establishes the first comprehensive defense mechanism against CoTA vulnerabilities in reinforcement learning-aligned reasoning systems, significantly advancing the security-functionality equilibrium for next-generation AI architectures.



## **50. Watch, Listen, Understand, Mislead: Tri-modal Adversarial Attacks on Short Videos for Content Appropriateness Evaluation**

cs.CV

Accepted as long paper, SVU Workshop at ICCV 2025

**SubmitDate**: 2025-07-16    [abs](http://arxiv.org/abs/2507.11968v1) [paper-pdf](http://arxiv.org/pdf/2507.11968v1)

**Authors**: Sahid Hossain Mustakim, S M Jishanul Islam, Ummay Maria Muna, Montasir Chowdhury, Mohammed Jawwadul Islam, Sadia Ahmmed, Tashfia Sikder, Syed Tasdid Azam Dhrubo, Swakkhar Shatabda

**Abstract**: Multimodal Large Language Models (MLLMs) are increasingly used for content moderation, yet their robustness in short-form video contexts remains underexplored. Current safety evaluations often rely on unimodal attacks, failing to address combined attack vulnerabilities. In this paper, we introduce a comprehensive framework for evaluating the tri-modal safety of MLLMs. First, we present the Short-Video Multimodal Adversarial (SVMA) dataset, comprising diverse short-form videos with human-guided synthetic adversarial attacks. Second, we propose ChimeraBreak, a novel tri-modal attack strategy that simultaneously challenges visual, auditory, and semantic reasoning pathways. Extensive experiments on state-of-the-art MLLMs reveal significant vulnerabilities with high Attack Success Rates (ASR). Our findings uncover distinct failure modes, showing model biases toward misclassifying benign or policy-violating content. We assess results using LLM-as-a-judge, demonstrating attack reasoning efficacy. Our dataset and findings provide crucial insights for developing more robust and safe MLLMs.



