# Latest Large Language Model Attack Papers
**update at 2025-07-24 10:55:58**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. LoX: Low-Rank Extrapolation Robustifies LLM Safety Against Fine-tuning**

cs.LG

**SubmitDate**: 2025-07-23    [abs](http://arxiv.org/abs/2506.15606v2) [paper-pdf](http://arxiv.org/pdf/2506.15606v2)

**Authors**: Gabriel J. Perin, Runjin Chen, Xuxi Chen, Nina S. T. Hirata, Zhangyang Wang, Junyuan Hong

**Abstract**: Large Language Models (LLMs) have become indispensable in real-world applications. However, their widespread adoption raises significant safety concerns, particularly in responding to socially harmful questions. Despite substantial efforts to improve model safety through alignment, aligned models can still have their safety protections undermined by subsequent fine-tuning - even when the additional training data appears benign. In this paper, we empirically demonstrate that this vulnerability stems from the sensitivity of safety-critical low-rank subspaces in LLM parameters to fine-tuning. Building on this insight, we propose a novel training-free method, termed Low-Rank Extrapolation (LoX), to enhance safety robustness by extrapolating the safety subspace of an aligned LLM. Our experimental results confirm the effectiveness of LoX, demonstrating significant improvements in robustness against both benign and malicious fine-tuning attacks while preserving the model's adaptability to new tasks. For instance, LoX leads to 11% to 54% absolute reductions in attack success rates (ASR) facing benign or malicious fine-tuning attacks. By investigating the ASR landscape of parameters, we attribute the success of LoX to that the extrapolation moves LLM parameters to a flatter zone, thereby less sensitive to perturbations. The code is available at github.com/VITA-Group/LoX.



## **2. MEF: A Capability-Aware Multi-Encryption Framework for Evaluating Vulnerabilities in Black-Box Large Language Models**

cs.CL

**SubmitDate**: 2025-07-23    [abs](http://arxiv.org/abs/2505.23404v4) [paper-pdf](http://arxiv.org/pdf/2505.23404v4)

**Authors**: Mingyu Yu, Wei Wang, Yanjie Wei, Sujuan Qin, Fei Gao, Wenmin Li

**Abstract**: Recent advancements in adversarial jailbreak attacks have exposed critical vulnerabilities in Large Language Models (LLMs), enabling the circumvention of alignment safeguards through increasingly sophisticated prompt manipulations. Based on our experiments, we found that the effectiveness of jailbreak strategies is influenced by the comprehension ability of the attacked LLM. Building on this insight, we propose a capability-aware Multi-Encryption Framework (MEF) for evaluating vulnerabilities in black-box LLMs. Specifically, MEF first categorizes the comprehension ability level of the LLM, then applies different strategies accordingly: For models with limited comprehension ability, MEF adopts the Fu+En1 strategy, which integrates layered semantic mutations with an encryption technique, more effectively contributing to evasion of the LLM's defenses at the input and inference stages. For models with strong comprehension ability, MEF uses a more complex Fu+En1+En2 strategy, in which additional dual-ended encryption techniques are applied to the LLM's responses, further contributing to evasion of the LLM's defenses at the output stage. Experimental results demonstrate the effectiveness of our approach, achieving attack success rates of 98.9% on GPT-4o (29 May 2025 release) and 99.8% on GPT-4.1 (8 July 2025 release). Our work contributes to a deeper understanding of the vulnerabilities in current LLM alignment mechanisms.



## **3. Tab-MIA: A Benchmark Dataset for Membership Inference Attacks on Tabular Data in LLMs**

cs.CR

**SubmitDate**: 2025-07-23    [abs](http://arxiv.org/abs/2507.17259v1) [paper-pdf](http://arxiv.org/pdf/2507.17259v1)

**Authors**: Eyal German, Sagiv Antebi, Daniel Samira, Asaf Shabtai, Yuval Elovici

**Abstract**: Large language models (LLMs) are increasingly trained on tabular data, which, unlike unstructured text, often contains personally identifiable information (PII) in a highly structured and explicit format. As a result, privacy risks arise, since sensitive records can be inadvertently retained by the model and exposed through data extraction or membership inference attacks (MIAs). While existing MIA methods primarily target textual content, their efficacy and threat implications may differ when applied to structured data, due to its limited content, diverse data types, unique value distributions, and column-level semantics. In this paper, we present Tab-MIA, a benchmark dataset for evaluating MIAs on tabular data in LLMs and demonstrate how it can be used. Tab-MIA comprises five data collections, each represented in six different encoding formats. Using our Tab-MIA benchmark, we conduct the first evaluation of state-of-the-art MIA methods on LLMs finetuned with tabular data across multiple encoding formats. In the evaluation, we analyze the memorization behavior of pretrained LLMs on structured data derived from Wikipedia tables. Our findings show that LLMs memorize tabular data in ways that vary across encoding formats, making them susceptible to extraction via MIAs. Even when fine-tuned for as few as three epochs, models exhibit high vulnerability, with AUROC scores approaching 90% in most cases. Tab-MIA enables systematic evaluation of these risks and provides a foundation for developing privacy-preserving methods for tabular data in LLMs.



## **4. LLM4MEA: Data-free Model Extraction Attacks on Sequential Recommenders via Large Language Models**

cs.IR

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2507.16969v1) [paper-pdf](http://arxiv.org/pdf/2507.16969v1)

**Authors**: Shilong Zhao, Fei Sun, Kaike Zhang, Shaoling Jing, Du Su, Zhichao Shi, Zhiyi Yin, Huawei Shen, Xueqi Cheng

**Abstract**: Recent studies have demonstrated the vulnerability of sequential recommender systems to Model Extraction Attacks (MEAs). MEAs collect responses from recommender systems to replicate their functionality, enabling unauthorized deployments and posing critical privacy and security risks. Black-box attacks in prior MEAs are ineffective at exposing recommender system vulnerabilities due to random sampling in data selection, which leads to misaligned synthetic and real-world distributions. To overcome this limitation, we propose LLM4MEA, a novel model extraction method that leverages Large Language Models (LLMs) as human-like rankers to generate data. It generates data through interactions between the LLM ranker and target recommender system. In each interaction, the LLM ranker analyzes historical interactions to understand user behavior, and selects items from recommendations with consistent preferences to extend the interaction history, which serves as training data for MEA. Extensive experiments demonstrate that LLM4MEA significantly outperforms existing approaches in data quality and attack performance, reducing the divergence between synthetic and real-world data by up to 64.98% and improving MEA performance by 44.82% on average. From a defensive perspective, we propose a simple yet effective defense strategy and identify key hyperparameters of recommender systems that can mitigate the risk of MEAs.



## **5. More is Less: The Pitfalls of Multi-Model Synthetic Preference Data in DPO Safety Alignment**

cs.AI

This version includes updated results and expanded discussion

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2504.02193v2) [paper-pdf](http://arxiv.org/pdf/2504.02193v2)

**Authors**: Yifan Wang, Runjin Chen, Bolian Li, David Cho, Yihe Deng, Ruqi Zhang, Tianlong Chen, Zhangyang Wang, Ananth Grama, Junyuan Hong

**Abstract**: Aligning large language models (LLMs) with human values is an increasingly critical step in post-training. Direct Preference Optimization (DPO) has emerged as a simple, yet effective alternative to reinforcement learning from human feedback (RLHF). Synthetic preference data with its low cost and high quality enable effective alignment through single- or multi-model generated preference data. Our study reveals a striking, safety-specific phenomenon associated with DPO alignment: Although multi-model generated data enhances performance on general tasks (ARC, Hellaswag, MMLU, TruthfulQA, Winogrande) by providing diverse responses, it also tends to facilitate reward hacking during training. This can lead to a high attack success rate (ASR) when models encounter jailbreaking prompts. The issue is particularly pronounced when employing stronger models like GPT-4o or larger models in the same family to generate chosen responses paired with target model self-generated rejected responses, resulting in dramatically poorer safety outcomes. Furthermore, with respect to safety, using solely self-generated responses (single-model generation) for both chosen and rejected pairs significantly outperforms configurations that incorporate responses from stronger models, whether used directly as chosen data or as part of a multi-model response pool. We demonstrate that multi-model preference data exhibits high linear separability between chosen and rejected responses, which allows models to exploit superficial cues rather than internalizing robust safety constraints. Our experiments, conducted on models from the Llama, Mistral, and Qwen families, consistently validate these findings.



## **6. When LLMs Copy to Think: Uncovering Copy-Guided Attacks in Reasoning LLMs**

cs.CR

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2507.16773v1) [paper-pdf](http://arxiv.org/pdf/2507.16773v1)

**Authors**: Yue Li, Xiao Li, Hao Wu, Yue Zhang, Fengyuan Xu, Xiuzhen Cheng, Sheng Zhong

**Abstract**: Large Language Models (LLMs) have become integral to automated code analysis, enabling tasks such as vulnerability detection and code comprehension. However, their integration introduces novel attack surfaces. In this paper, we identify and investigate a new class of prompt-based attacks, termed Copy-Guided Attacks (CGA), which exploit the inherent copying tendencies of reasoning-capable LLMs. By injecting carefully crafted triggers into external code snippets, adversaries can induce the model to replicate malicious content during inference. This behavior enables two classes of vulnerabilities: inference length manipulation, where the model generates abnormally short or excessively long reasoning traces; and inference result manipulation, where the model produces misleading or incorrect conclusions. We formalize CGA as an optimization problem and propose a gradient-based approach to synthesize effective triggers. Empirical evaluation on state-of-the-art reasoning LLMs shows that CGA reliably induces infinite loops, premature termination, false refusals, and semantic distortions in code analysis tasks. While highly effective in targeted settings, we observe challenges in generalizing CGA across diverse prompts due to computational constraints, posing an open question for future research. Our findings expose a critical yet underexplored vulnerability in LLM-powered development pipelines and call for urgent advances in prompt-level defense mechanisms.



## **7. From Text to Actionable Intelligence: Automating STIX Entity and Relationship Extraction**

cs.CR

This paper is accepted at RAID 2025

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2507.16576v1) [paper-pdf](http://arxiv.org/pdf/2507.16576v1)

**Authors**: Ahmed Lekssays, Husrev Taha Sencar, Ting Yu

**Abstract**: Sharing methods of attack and their effectiveness is a cornerstone of building robust defensive systems. Threat analysis reports, produced by various individuals and organizations, play a critical role in supporting security operations and combating emerging threats. To enhance the timeliness and automation of threat intelligence sharing, several standards have been established, with the Structured Threat Information Expression (STIX) framework emerging as one of the most widely adopted. However, generating STIX-compatible data from unstructured security text remains a largely manual, expert-driven process. To address this challenge, we introduce AZERG, a tool designed to assist security analysts in automatically generating structured STIX representations. To achieve this, we adapt general-purpose large language models for the specific task of extracting STIX-formatted threat data. To manage the complexity, the task is divided into four subtasks: entity detection (T1), entity type identification (T2), related pair detection (T3), and relationship type identification (T4). We apply task-specific fine-tuning to accurately extract relevant entities and infer their relationships in accordance with the STIX specification. To address the lack of training data, we compiled a comprehensive dataset with 4,011 entities and 2,075 relationships extracted from 141 full threat analysis reports, all annotated in alignment with the STIX standard. Our models achieved F1-scores of 84.43% for T1, 88.49% for T2, 95.47% for T3, and 84.60% for T4 in real-world scenarios. We validated their performance against a range of open- and closed-parameter models, as well as state-of-the-art methods, demonstrating improvements of 2-25% across tasks.



## **8. Depth Gives a False Sense of Privacy: LLM Internal States Inversion**

cs.CR

Accepted by USENIX Security 2025. Please cite this paper as "Tian  Dong, Yan Meng, Shaofeng Li, Guoxing Chen, Zhen Liu, Haojin Zhu. Depth Gives  a False Sense of Privacy: LLM Internal States Inversion. In the 34th USENIX  Security Symposium (USENIX Security '25)."

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2507.16372v1) [paper-pdf](http://arxiv.org/pdf/2507.16372v1)

**Authors**: Tian Dong, Yan Meng, Shaofeng Li, Guoxing Chen, Zhen Liu, Haojin Zhu

**Abstract**: Large Language Models (LLMs) are increasingly integrated into daily routines, yet they raise significant privacy and safety concerns. Recent research proposes collaborative inference, which outsources the early-layer inference to ensure data locality, and introduces model safety auditing based on inner neuron patterns. Both techniques expose the LLM's Internal States (ISs), which are traditionally considered irreversible to inputs due to optimization challenges and the highly abstract representations in deep layers. In this work, we challenge this assumption by proposing four inversion attacks that significantly improve the semantic similarity and token matching rate of inverted inputs. Specifically, we first develop two white-box optimization-based attacks tailored for low-depth and high-depth ISs. These attacks avoid local minima convergence, a limitation observed in prior work, through a two-phase inversion process. Then, we extend our optimization attack under more practical black-box weight access by leveraging the transferability between the source and the derived LLMs. Additionally, we introduce a generation-based attack that treats inversion as a translation task, employing an inversion model to reconstruct inputs. Extensive evaluation of short and long prompts from medical consulting and coding assistance datasets and 6 LLMs validates the effectiveness of our inversion attacks. Notably, a 4,112-token long medical consulting prompt can be nearly perfectly inverted with 86.88 F1 token matching from the middle layer of Llama-3 model. Finally, we evaluate four practical defenses that we found cannot perfectly prevent ISs inversion and draw conclusions for future mitigation design.



## **9. Can Indirect Prompt Injection Attacks Be Detected and Removed?**

cs.CR

To Appear in ACL 2025

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2502.16580v3) [paper-pdf](http://arxiv.org/pdf/2502.16580v3)

**Authors**: Yulin Chen, Haoran Li, Yuan Sui, Yufei He, Yue Liu, Yangqiu Song, Bryan Hooi

**Abstract**: Prompt injection attacks manipulate large language models (LLMs) by misleading them to deviate from the original input instructions and execute maliciously injected instructions, because of their instruction-following capabilities and inability to distinguish between the original input instructions and maliciously injected instructions. To defend against such attacks, recent studies have developed various detection mechanisms. If we restrict ourselves specifically to works which perform detection rather than direct defense, most of them focus on direct prompt injection attacks, while there are few works for the indirect scenario, where injected instructions are indirectly from external tools, such as a search engine. Moreover, current works mainly investigate injection detection methods and pay less attention to the post-processing method that aims to mitigate the injection after detection. In this paper, we investigate the feasibility of detecting and removing indirect prompt injection attacks, and we construct a benchmark dataset for evaluation. For detection, we assess the performance of existing LLMs and open-source detection models, and we further train detection models using our crafted training datasets. For removal, we evaluate two intuitive methods: (1) the segmentation removal method, which segments the injected document and removes parts containing injected instructions, and (2) the extraction removal method, which trains an extraction model to identify and remove injected instructions.



## **10. ShadowCode: Towards (Automatic) External Prompt Injection Attack against Code LLMs**

cs.CR

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2407.09164v6) [paper-pdf](http://arxiv.org/pdf/2407.09164v6)

**Authors**: Yuchen Yang, Yiming Li, Hongwei Yao, Bingrun Yang, Yiling He, Tianwei Zhang, Dacheng Tao, Zhan Qin

**Abstract**: Recent advancements have led to the widespread adoption of code-oriented large language models (Code LLMs) for programming tasks. Despite their success in deployment, their security research is left far behind. This paper introduces a new attack paradigm: (automatic) external prompt injection against Code LLMs, where attackers generate concise, non-functional induced perturbations and inject them within a victim's code context. These induced perturbations can be disseminated through commonly used dependencies (e.g., packages or RAG's knowledge base), manipulating Code LLMs to achieve malicious objectives during the code completion process. Compared to existing attacks, this method is more realistic and threatening: it does not necessitate control over the model's training process, unlike backdoor attacks, and can achieve specific malicious objectives that are challenging for adversarial attacks. Furthermore, we propose ShadowCode, a simple yet effective method that automatically generates induced perturbations based on code simulation to achieve effective and stealthy external prompt injection. ShadowCode designs its perturbation optimization objectives by simulating realistic code contexts and employs a greedy optimization approach with two enhancement modules: forward reasoning enhancement and keyword-based perturbation design. We evaluate our method across 13 distinct malicious objectives, generating 31 threat cases spanning three popular programming languages. Our results demonstrate that ShadowCode successfully attacks three representative open-source Code LLMs (achieving up to a 97.9% attack success rate) and two mainstream commercial Code LLM-integrated applications (with over 90% attack success rate) across all threat cases, using only a 12-token non-functional induced perturbation. The code is available at https://github.com/LianPing-cyber/ShadowCodeEPI.



## **11. Defense Against Prompt Injection Attack by Leveraging Attack Techniques**

cs.CR

To Appear in ACL 2025

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2411.00459v5) [paper-pdf](http://arxiv.org/pdf/2411.00459v5)

**Authors**: Yulin Chen, Haoran Li, Zihao Zheng, Yangqiu Song, Dekai Wu, Bryan Hooi

**Abstract**: With the advancement of technology, large language models (LLMs) have achieved remarkable performance across various natural language processing (NLP) tasks, powering LLM-integrated applications like Microsoft Copilot. However, as LLMs continue to evolve, new vulnerabilities, especially prompt injection attacks arise. These attacks trick LLMs into deviating from the original input instructions and executing the attacker's instructions injected in data content, such as retrieved results. Recent attack methods leverage LLMs' instruction-following abilities and their inabilities to distinguish instructions injected in the data content, and achieve a high attack success rate (ASR). When comparing the attack and defense methods, we interestingly find that they share similar design goals, of inducing the model to ignore unwanted instructions and instead to execute wanted instructions. Therefore, we raise an intuitive question: Could these attack techniques be utilized for defensive purposes? In this paper, we invert the intention of prompt injection methods to develop novel defense methods based on previous training-free attack methods, by repeating the attack process but with the original input instruction rather than the injected instruction. Our comprehensive experiments demonstrate that our defense techniques outperform existing training-free defense approaches, achieving state-of-the-art results.



## **12. CompLeak: Deep Learning Model Compression Exacerbates Privacy Leakage**

cs.CR

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2507.16872v1) [paper-pdf](http://arxiv.org/pdf/2507.16872v1)

**Authors**: Na Li, Yansong Gao, Hongsheng Hu, Boyu Kuang, Anmin Fu

**Abstract**: Model compression is crucial for minimizing memory storage and accelerating inference in deep learning (DL) models, including recent foundation models like large language models (LLMs). Users can access different compressed model versions according to their resources and budget. However, while existing compression operations primarily focus on optimizing the trade-off between resource efficiency and model performance, the privacy risks introduced by compression remain overlooked and insufficiently understood.   In this work, through the lens of membership inference attack (MIA), we propose CompLeak, the first privacy risk evaluation framework examining three widely used compression configurations that are pruning, quantization, and weight clustering supported by the commercial model compression framework of Google's TensorFlow-Lite (TF-Lite) and Facebook's PyTorch Mobile. CompLeak has three variants, given available access to the number of compressed models and original model. CompLeakNR starts by adopting existing MIA methods to attack a single compressed model, and identifies that different compressed models influence members and non-members differently. When the original model and one compressed model are available, CompLeakSR leverages the compressed model as a reference to the original model and uncovers more privacy by combining meta information (e.g., confidence vector) from both models. When multiple compressed models are available with/without accessing the original model, CompLeakMR innovatively exploits privacy leakage info from multiple compressed versions to substantially signify the overall privacy leakage. We conduct extensive experiments on seven diverse model architectures (from ResNet to foundation models of BERT and GPT-2), and six image and textual benchmark datasets.



## **13. OMNISEC: LLM-Driven Provenance-based Intrusion Detection via Retrieval-Augmented Behavior Prompting**

cs.CR

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2503.03108v4) [paper-pdf](http://arxiv.org/pdf/2503.03108v4)

**Authors**: Wenrui Cheng, Tiantian Zhu, Shunan Jing, Jian-Ping Mei, Mingjun Ma, Jiaobo Jin, Zhengqiu Weng

**Abstract**: Recently, Provenance-based Intrusion Detection Systems (PIDSes) have been widely used for endpoint threat analysis. These studies can be broadly categorized into rule-based detection systems and learning-based detection systems. Among these, due to the evolution of attack techniques, rules cannot dynamically model all the characteristics of attackers. As a result, such systems often face false negatives. Learning-based detection systems are further divided into supervised learning and anomaly detection. The scarcity of attack samples hinders the usability and effectiveness of supervised learning-based detection systems in practical applications. Anomaly-based detection systems face a massive false positive problem because they cannot distinguish between changes in normal behavior and real attack behavior. The alert results of detection systems are closely related to the manual labor costs of subsequent security analysts. To reduce manual analysis time, we propose OMNISEC, which applies large language models (LLMs) to anomaly-based intrusion detection systems via retrieval-augmented behavior prompting. OMNISEC can identify abnormal nodes and corresponding abnormal events by constructing suspicious nodes and rare paths. By combining two external knowledge bases, OMNISEC uses Retrieval Augmented Generation (RAG) to enable the LLM to determine whether abnormal behavior is a real attack. Finally, OMNISEC can reconstruct the attack graph and restore the complete attack behavior chain of the attacker's intrusion. Experimental results show that OMNISEC outperforms state-of-the-art methods on public benchmark datasets.



## **14. Talking Like a Phisher: LLM-Based Attacks on Voice Phishing Classifiers**

cs.CR

Accepted by EAI ICDF2C 2025

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2507.16291v1) [paper-pdf](http://arxiv.org/pdf/2507.16291v1)

**Authors**: Wenhao Li, Selvakumar Manickam, Yung-wey Chong, Shankar Karuppayah

**Abstract**: Voice phishing (vishing) remains a persistent threat in cybersecurity, exploiting human trust through persuasive speech. While machine learning (ML)-based classifiers have shown promise in detecting malicious call transcripts, they remain vulnerable to adversarial manipulations that preserve semantic content. In this study, we explore a novel attack vector where large language models (LLMs) are leveraged to generate adversarial vishing transcripts that evade detection while maintaining deceptive intent. We construct a systematic attack pipeline that employs prompt engineering and semantic obfuscation to transform real-world vishing scripts using four commercial LLMs. The generated transcripts are evaluated against multiple ML classifiers trained on a real-world Korean vishing dataset (KorCCViD) with statistical testing. Our experiments reveal that LLM-generated transcripts are both practically and statistically effective against ML-based classifiers. In particular, transcripts crafted by GPT-4o significantly reduce classifier accuracy (by up to 30.96%) while maintaining high semantic similarity, as measured by BERTScore. Moreover, these attacks are both time-efficient and cost-effective, with average generation times under 9 seconds and negligible financial cost per query. The results underscore the pressing need for more resilient vishing detection frameworks and highlight the imperative for LLM providers to enforce stronger safeguards against prompt misuse in adversarial social engineering contexts.



## **15. Quality Text, Robust Vision: The Role of Language in Enhancing Visual Robustness of Vision-Language Models**

cs.CV

ACMMM 2025 Accepted

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2507.16257v1) [paper-pdf](http://arxiv.org/pdf/2507.16257v1)

**Authors**: Futa Waseda, Saku Sugawara, Isao Echizen

**Abstract**: Defending pre-trained vision-language models (VLMs), such as CLIP, against adversarial attacks is crucial, as these models are widely used in diverse zero-shot tasks, including image classification. However, existing adversarial training (AT) methods for robust fine-tuning largely overlook the role of language in enhancing visual robustness. Specifically, (1) supervised AT methods rely on short texts (e.g., class labels) to generate adversarial perturbations, leading to overfitting to object classes in the training data, and (2) unsupervised AT avoids this overfitting but remains suboptimal against practical text-guided adversarial attacks due to its lack of semantic guidance. To address these limitations, we propose Quality Text-guided Adversarial Fine-Tuning (QT-AFT), which leverages high-quality captions during training to guide adversarial examples away from diverse semantics present in images. This enables the visual encoder to robustly recognize a broader range of image features even under adversarial noise, thereby enhancing robustness across diverse downstream tasks. QT-AFT overcomes the key weaknesses of prior methods -- overfitting in supervised AT and lack of semantic awareness in unsupervised AT -- achieving state-of-the-art zero-shot adversarial robustness and clean accuracy, evaluated across 16 zero-shot datasets. Furthermore, our comprehensive study uncovers several key insights into the role of language in enhancing vision robustness; for example, describing object properties in addition to object names further enhances zero-shot robustness. Our findings point to an urgent direction for future work -- centering high-quality linguistic supervision in robust visual representation learning.



## **16. BackdoorDM: A Comprehensive Benchmark for Backdoor Learning on Diffusion Model**

cs.CR

**SubmitDate**: 2025-07-21    [abs](http://arxiv.org/abs/2502.11798v2) [paper-pdf](http://arxiv.org/pdf/2502.11798v2)

**Authors**: Weilin Lin, Nanjun Zhou, Yanyun Wang, Jianze Li, Hui Xiong, Li Liu

**Abstract**: Backdoor learning is a critical research topic for understanding the vulnerabilities of deep neural networks. While the diffusion model (DM) has been broadly deployed in public over the past few years, the understanding of its backdoor vulnerability is still in its infancy compared to the extensive studies in discriminative models. Recently, many different backdoor attack and defense methods have been proposed for DMs, but a comprehensive benchmark for backdoor learning on DMs is still lacking. This absence makes it difficult to conduct fair comparisons and thorough evaluations of the existing approaches, thus hindering future research progress. To address this issue, we propose \textit{BackdoorDM}, the first comprehensive benchmark designed for backdoor learning on DMs. It comprises nine state-of-the-art (SOTA) attack methods, four SOTA defense strategies, and three useful visualization analysis tools. We first systematically classify and formulate the existing literature in a unified framework, focusing on three different backdoor attack types and five backdoor target types, which are restricted to a single type in discriminative models. Then, we systematically summarize the evaluation metrics for each type and propose a unified backdoor evaluation method based on multimodal large language model (MLLM). Finally, we conduct a comprehensive evaluation and highlight several important conclusions. We believe that BackdoorDM will help overcome current barriers and contribute to building a trustworthy artificial intelligence generated content (AIGC) community. The codes are released in https://github.com/linweiii/BackdoorDM.



## **17. Multi-Stage Prompt Inference Attacks on Enterprise LLM Systems**

cs.CR

26 pages

**SubmitDate**: 2025-07-21    [abs](http://arxiv.org/abs/2507.15613v1) [paper-pdf](http://arxiv.org/pdf/2507.15613v1)

**Authors**: Andrii Balashov, Olena Ponomarova, Xiaohua Zhai

**Abstract**: Large Language Models (LLMs) deployed in enterprise settings (e.g., as Microsoft 365 Copilot) face novel security challenges. One critical threat is prompt inference attacks: adversaries chain together seemingly benign prompts to gradually extract confidential data. In this paper, we present a comprehensive study of multi-stage prompt inference attacks in an enterprise LLM context. We simulate realistic attack scenarios where an attacker uses mild-mannered queries and indirect prompt injections to exploit an LLM integrated with private corporate data. We develop a formal threat model for these multi-turn inference attacks and analyze them using probability theory, optimization frameworks, and information-theoretic leakage bounds. The attacks are shown to reliably exfiltrate sensitive information from the LLM's context (e.g., internal SharePoint documents or emails), even when standard safety measures are in place.   We propose and evaluate defenses to counter such attacks, including statistical anomaly detection, fine-grained access control, prompt sanitization techniques, and architectural modifications to LLM deployment. Each defense is supported by mathematical analysis or experimental simulation. For example, we derive bounds on information leakage under differential privacy-based training and demonstrate an anomaly detection method that flags multi-turn attacks with high AUC. We also introduce an approach called "spotlighting" that uses input transformations to isolate untrusted prompt content, reducing attack success by an order of magnitude. Finally, we provide a formal proof of concept and empirical validation for a combined defense-in-depth strategy. Our work highlights that securing LLMs in enterprise settings requires moving beyond single-turn prompt filtering toward a holistic, multi-stage perspective on both attacks and defenses.



## **18. PiMRef: Detecting and Explaining Ever-evolving Spear Phishing Emails with Knowledge Base Invariants**

cs.CR

**SubmitDate**: 2025-07-21    [abs](http://arxiv.org/abs/2507.15393v1) [paper-pdf](http://arxiv.org/pdf/2507.15393v1)

**Authors**: Ruofan Liu, Yun Lin, Silas Yeo Shuen Yu, Xiwen Teoh, Zhenkai Liang, Jin Song Dong

**Abstract**: Phishing emails are a critical component of the cybercrime kill chain due to their wide reach and low cost. Their ever-evolving nature renders traditional rule-based and feature-engineered detectors ineffective in the ongoing arms race between attackers and defenders. The rise of large language models (LLMs) further exacerbates the threat, enabling attackers to craft highly convincing phishing emails at minimal cost.   This work demonstrates that LLMs can generate psychologically persuasive phishing emails tailored to victim profiles, successfully bypassing nearly all commercial and academic detectors. To defend against such threats, we propose PiMRef, the first reference-based phishing email detector that leverages knowledge-based invariants. Our core insight is that persuasive phishing emails often contain disprovable identity claims, which contradict real-world facts. PiMRef reframes phishing detection as an identity fact-checking task. Given an email, PiMRef (i) extracts the sender's claimed identity, (ii) verifies the legitimacy of the sender's domain against a predefined knowledge base, and (iii) detects call-to-action prompts that push user engagement. Contradictory claims are flagged as phishing indicators and serve as human-understandable explanations.   Compared to existing methods such as D-Fence, HelpHed, and ChatSpamDetector, PiMRef boosts precision by 8.8% with no loss in recall on standard benchmarks like Nazario and PhishPot. In a real-world evaluation of 10,183 emails across five university accounts over three years, PiMRef achieved 92.1% precision, 87.9% recall, and a median runtime of 0.05s, outperforming the state-of-the-art in both effectiveness and efficiency.



## **19. Transfer Attack for Bad and Good: Explain and Boost Adversarial Transferability across Multimodal Large Language Models**

cs.CV

This paper is accepted by ACM MM 2025

**SubmitDate**: 2025-07-21    [abs](http://arxiv.org/abs/2405.20090v5) [paper-pdf](http://arxiv.org/pdf/2405.20090v5)

**Authors**: Hao Cheng, Erjia Xiao, Jiayan Yang, Jinhao Duan, Yichi Wang, Jiahang Cao, Qiang Zhang, Le Yang, Kaidi Xu, Jindong Gu, Renjing Xu

**Abstract**: Multimodal Large Language Models (MLLMs) demonstrate exceptional performance in cross-modality interaction, yet they also suffer adversarial vulnerabilities. In particular, the transferability of adversarial examples remains an ongoing challenge. In this paper, we specifically analyze the manifestation of adversarial transferability among MLLMs and identify the key factors that influence this characteristic. We discover that the transferability of MLLMs exists in cross-LLM scenarios with the same vision encoder and indicate \underline{\textit{two key Factors}} that may influence transferability. We provide two semantic-level data augmentation methods, Adding Image Patch (AIP) and Typography Augment Transferability Method (TATM), which boost the transferability of adversarial examples across MLLMs. To explore the potential impact in the real world, we utilize two tasks that can have both negative and positive societal impacts: \ding{182} Harmful Content Insertion and \ding{183} Information Protection.



## **20. Scaling Decentralized Learning with FLock**

cs.LG

**SubmitDate**: 2025-07-21    [abs](http://arxiv.org/abs/2507.15349v1) [paper-pdf](http://arxiv.org/pdf/2507.15349v1)

**Authors**: Zehua Cheng, Rui Sun, Jiahao Sun, Yike Guo

**Abstract**: Fine-tuning the large language models (LLMs) are prevented by the deficiency of centralized control and the massive computing and communication overhead on the decentralized schemes. While the typical standard federated learning (FL) supports data privacy, the central server requirement creates a single point of attack and vulnerability to poisoning attacks. Generalizing the result in this direction to 70B-parameter models in the heterogeneous, trustless environments has turned out to be a huge, yet unbroken bottleneck. This paper introduces FLock, a decentralized framework for secure and efficient collaborative LLM fine-tuning. Integrating a blockchain-based trust layer with economic incentives, FLock replaces the central aggregator with a secure, auditable protocol for cooperation among untrusted parties. We present the first empirical validation of fine-tuning a 70B LLM in a secure, multi-domain, decentralized setting. Our experiments show the FLock framework defends against backdoor poisoning attacks that compromise standard FL optimizers and fosters synergistic knowledge transfer. The resulting models show a >68% reduction in adversarial attack success rates. The global model also demonstrates superior cross-domain generalization, outperforming models trained in isolation on their own specialized data.



## **21. From LLMs to MLLMs to Agents: A Survey of Emerging Paradigms in Jailbreak Attacks and Defenses within LLM Ecosystem**

cs.CR

**SubmitDate**: 2025-07-20    [abs](http://arxiv.org/abs/2506.15170v2) [paper-pdf](http://arxiv.org/pdf/2506.15170v2)

**Authors**: Yanxu Mao, Tiehan Cui, Peipei Liu, Datao You, Hongsong Zhu

**Abstract**: Large language models (LLMs) are rapidly evolving from single-modal systems to multimodal LLMs and intelligent agents, significantly expanding their capabilities while introducing increasingly severe security risks. This paper presents a systematic survey of the growing complexity of jailbreak attacks and corresponding defense mechanisms within the expanding LLM ecosystem. We first trace the developmental trajectory from LLMs to MLLMs and Agents, highlighting the core security challenges emerging at each stage. Next, we categorize mainstream jailbreak techniques from both the attack impact and visibility perspectives, and provide a comprehensive analysis of representative attack methods, related datasets, and evaluation metrics. On the defense side, we organize existing strategies based on response timing and technical approach, offering a structured understanding of their applicability and implementation. Furthermore, we identify key limitations in existing surveys, such as insufficient attention to agent-specific security issues, the absence of a clear taxonomy for hybrid jailbreak methods, a lack of detailed analysis of experimental setups, and outdated coverage of recent advancements. To address these limitations, we provide an updated synthesis of recent work and outline future research directions in areas such as dataset construction, evaluation framework optimization, and strategy generalization. Our study seeks to enhance the understanding of jailbreak mechanisms and facilitate the advancement of more resilient and adaptive defense strategies in the context of ever more capable LLMs.



## **22. Byzantine-Robust Decentralized Coordination of LLM Agents**

cs.DC

**SubmitDate**: 2025-07-20    [abs](http://arxiv.org/abs/2507.14928v1) [paper-pdf](http://arxiv.org/pdf/2507.14928v1)

**Authors**: Yongrae Jo, Chanik Park

**Abstract**: Collaboration among multiple large language model (LLM) agents is a promising approach to overcome inherent limitations of single-agent systems, such as hallucinations and single points of failure. As LLM agents are increasingly deployed on open blockchain platforms, multi-agent systems capable of tolerating malicious (Byzantine) agents have become essential.   Recent Byzantine-robust multi-agent systems typically rely on leader-driven coordination, which suffers from two major drawbacks. First, they are inherently vulnerable to targeted attacks against the leader. If consecutive leaders behave maliciously, the system repeatedly fails to achieve consensus, forcing new consensus rounds, which is particularly costly given the high latency of LLM invocations. Second, an underperforming proposal from the leader can be accepted as the final answer even when higher-quality alternatives are available, as existing methods finalize the leader's proposal once it receives a quorum of votes.   To address these issues, we propose DecentLLMs, a novel decentralized consensus approach for multi-agent LLM systems, where worker agents generate answers concurrently and evaluator agents independently score and rank these answers to select the best available one. This decentralized architecture enables faster consensus despite the presence of Byzantine agents and consistently selects higher-quality answers through Byzantine-robust aggregation techniques.   Experimental results demonstrate that DecentLLMs effectively tolerates Byzantine agents and significantly improves the quality of selected answers.



## **23. Configurable multi-agent framework for scalable and realistic testing of llm-based agents**

cs.AI

**SubmitDate**: 2025-07-19    [abs](http://arxiv.org/abs/2507.14705v1) [paper-pdf](http://arxiv.org/pdf/2507.14705v1)

**Authors**: Sai Wang, Senthilnathan Subramanian, Mudit Sahni, Praneeth Gone, Lingjie Meng, Xiaochen Wang, Nicolas Ferradas Bertoli, Tingxian Cheng, Jun Xu

**Abstract**: Large-language-model (LLM) agents exhibit complex, context-sensitive behaviour that quickly renders static benchmarks and ad-hoc manual testing obsolete.   We present Neo, a configurable, multi-agent framework that automates realistic, multi-turn evaluation of LLM-based systems. Neo couples a Question Generation Agent and an Evaluation Agent through a shared context-hub, allowing domain prompts, scenario controls and dynamic feedback to be composed modularly. Test inputs are sampled from a probabilistic state model spanning dialogue flow, user intent and emotional tone, enabling diverse, human-like conversations that adapt after every turn.   Applied to a production-grade Seller Financial Assistant chatbot, Neo (i) uncovered edge-case failures across five attack categories with a 3.3% break rate close to the 5.8% achieved by expert human red-teamers, and (ii) delivered 10-12X higher throughput, generating 180 coherent test questions in around 45 mins versus 16h of human effort. Beyond security probing, Neo's stochastic policies balanced topic coverage and conversational depth, yielding broader behavioural exploration than manually crafted scripts.   Neo therefore lays a foundation for scalable, self-evolving LLM QA: its agent interfaces, state controller and feedback loops are model-agnostic and extensible to richer factual-grounding and policy-compliance checks. We release the framework to facilitate reproducible, high-fidelity testing of emerging agentic systems.



## **24. Blackbox Dataset Inference for LLM**

cs.CR

**SubmitDate**: 2025-07-18    [abs](http://arxiv.org/abs/2507.03619v2) [paper-pdf](http://arxiv.org/pdf/2507.03619v2)

**Authors**: Ruikai Zhou, Kang Yang, Xun Chen, Wendy Hui Wang, Guanhong Tao, Jun Xu

**Abstract**: Today, the training of large language models (LLMs) can involve personally identifiable information and copyrighted material, incurring dataset misuse. To mitigate the problem of dataset misuse, this paper explores \textit{dataset inference}, which aims to detect if a suspect model $\mathcal{M}$ used a victim dataset $\mathcal{D}$ in training. Previous research tackles dataset inference by aggregating results of membership inference attacks (MIAs) -- methods to determine whether individual samples are a part of the training dataset. However, restricted by the low accuracy of MIAs, previous research mandates grey-box access to $\mathcal{M}$ to get intermediate outputs (probabilities, loss, perplexity, etc.) for obtaining satisfactory results. This leads to reduced practicality, as LLMs, especially those deployed for profits, have limited incentives to return the intermediate outputs.   In this paper, we propose a new method of dataset inference with only black-box access to the target model (i.e., assuming only the text-based responses of the target model are available). Our method is enabled by two sets of locally built reference models, one set involving $\mathcal{D}$ in training and the other not. By measuring which set of reference model $\mathcal{M}$ is closer to, we determine if $\mathcal{M}$ used $\mathcal{D}$ for training. Evaluations of real-world LLMs in the wild show that our method offers high accuracy in all settings and presents robustness against bypassing attempts.



## **25. From Words to Collisions: LLM-Guided Evaluation and Adversarial Generation of Safety-Critical Driving Scenarios**

cs.AI

Final Version and Paper Accepted at IEEE ITSC 2025

**SubmitDate**: 2025-07-18    [abs](http://arxiv.org/abs/2502.02145v4) [paper-pdf](http://arxiv.org/pdf/2502.02145v4)

**Authors**: Yuan Gao, Mattia Piccinini, Korbinian Moller, Amr Alanwar, Johannes Betz

**Abstract**: Ensuring the safety of autonomous vehicles requires virtual scenario-based testing, which depends on the robust evaluation and generation of safety-critical scenarios. So far, researchers have used scenario-based testing frameworks that rely heavily on handcrafted scenarios as safety metrics. To reduce the effort of human interpretation and overcome the limited scalability of these approaches, we combine Large Language Models (LLMs) with structured scenario parsing and prompt engineering to automatically evaluate and generate safety-critical driving scenarios. We introduce Cartesian and Ego-centric prompt strategies for scenario evaluation, and an adversarial generation module that modifies trajectories of risk-inducing vehicles (ego-attackers) to create critical scenarios. We validate our approach using a 2D simulation framework and multiple pre-trained LLMs. The results show that the evaluation module effectively detects collision scenarios and infers scenario safety. Meanwhile, the new generation module identifies high-risk agents and synthesizes realistic, safety-critical scenarios. We conclude that an LLM equipped with domain-informed prompting techniques can effectively evaluate and generate safety-critical driving scenarios, reducing dependence on handcrafted metrics. We release our open-source code and scenarios at: https://github.com/TUM-AVS/From-Words-to-Collisions.



## **26. TopicAttack: An Indirect Prompt Injection Attack via Topic Transition**

cs.CR

19 pages

**SubmitDate**: 2025-07-18    [abs](http://arxiv.org/abs/2507.13686v1) [paper-pdf](http://arxiv.org/pdf/2507.13686v1)

**Authors**: Yulin Chen, Haoran Li, Yuexin Li, Yue Liu, Yangqiu Song, Bryan Hooi

**Abstract**: Large language models (LLMs) have shown remarkable performance across a range of NLP tasks. However, their strong instruction-following capabilities and inability to distinguish instructions from data content make them vulnerable to indirect prompt injection attacks. In such attacks, instructions with malicious purposes are injected into external data sources, such as web documents. When LLMs retrieve this injected data through tools, such as a search engine and execute the injected instructions, they provide misled responses. Recent attack methods have demonstrated potential, but their abrupt instruction injection often undermines their effectiveness. Motivated by the limitations of existing attack methods, we propose TopicAttack, which prompts the LLM to generate a fabricated conversational transition prompt that gradually shifts the topic toward the injected instruction, making the injection smoother and enhancing the plausibility and success of the attack. Through comprehensive experiments, TopicAttack achieves state-of-the-art performance, with an attack success rate (ASR) over 90\% in most cases, even when various defense methods are applied. We further analyze its effectiveness by examining attention scores. We find that a higher injected-to-original attention ratio leads to a greater success probability, and our method achieves a much higher ratio than the baseline methods.



## **27. Invisible Textual Backdoor Attacks based on Dual-Trigger**

cs.CR

**SubmitDate**: 2025-07-18    [abs](http://arxiv.org/abs/2412.17531v3) [paper-pdf](http://arxiv.org/pdf/2412.17531v3)

**Authors**: Yang Hou, Qiuling Yue, Lujia Chai, Guozhao Liao, Wenbao Han, Wei Ou

**Abstract**: Backdoor attacks pose an important security threat to textual large language models. Exploring textual backdoor attacks not only helps reveal the potential security risks of models, but also promotes innovation and development of defense mechanisms. Currently, most textual backdoor attack methods are based on a single trigger. For example, inserting specific content into text as a trigger or changing the abstract text features to be a trigger. However, the adoption of this single-trigger mode makes the existing backdoor attacks subject to certain limitations: either they are easily identified by the existing defense strategies, or they have certain shortcomings in attack performance and in the construction of poisoned datasets. In order to solve these issues, a dual-trigger backdoor attack method is proposed in this paper. Specifically, we use two different attributes, syntax and mood (we use subjunctive mood as an example in this article), as two different triggers. It makes our backdoor attack method similar to a double landmine which can have completely different trigger conditions simultaneously. Therefore, this method not only improves the flexibility of trigger mode, but also enhances the robustness against defense detection. A large number of experimental results show that this method significantly outperforms the previous methods based on abstract features in attack performance, and achieves comparable attack performance (almost 100\% attack success rate) with the insertion-based method. In addition, in order to further improve the attack performance, we also give the construction method of the poisoned dataset.The code and data of this paper can be obtained at https://github.com/HoyaAm/Double-Landmines.



## **28. Paper Summary Attack: Jailbreaking LLMs through LLM Safety Papers**

cs.CL

**SubmitDate**: 2025-07-17    [abs](http://arxiv.org/abs/2507.13474v1) [paper-pdf](http://arxiv.org/pdf/2507.13474v1)

**Authors**: Liang Lin, Zhihao Xu, Xuehai Tang, Shi Liu, Biyu Zhou, Fuqing Zhu, Jizhong Han, Songlin Hu

**Abstract**: The safety of large language models (LLMs) has garnered significant research attention. In this paper, we argue that previous empirical studies demonstrate LLMs exhibit a propensity to trust information from authoritative sources, such as academic papers, implying new possible vulnerabilities. To verify this possibility, a preliminary analysis is designed to illustrate our two findings. Based on this insight, a novel jailbreaking method, Paper Summary Attack (\llmname{PSA}), is proposed. It systematically synthesizes content from either attack-focused or defense-focused LLM safety paper to construct an adversarial prompt template, while strategically infilling harmful query as adversarial payloads within predefined subsections. Extensive experiments show significant vulnerabilities not only in base LLMs, but also in state-of-the-art reasoning model like Deepseek-R1. PSA achieves a 97\% attack success rate (ASR) on well-aligned models like Claude3.5-Sonnet and an even higher 98\% ASR on Deepseek-R1. More intriguingly, our work has further revealed diametrically opposed vulnerability bias across different base models, and even between different versions of the same model, when exposed to either attack-focused or defense-focused papers. This phenomenon potentially indicates future research clues for both adversarial methodologies and safety alignment.Code is available at https://github.com/233liang/Paper-Summary-Attack



## **29. Automating Steering for Safe Multimodal Large Language Models**

cs.CL

Working in progress. 22 pages (8+ for main); 25 figures; 1 table

**SubmitDate**: 2025-07-17    [abs](http://arxiv.org/abs/2507.13255v1) [paper-pdf](http://arxiv.org/pdf/2507.13255v1)

**Authors**: Lyucheng Wu, Mengru Wang, Ziwen Xu, Tri Cao, Nay Oo, Bryan Hooi, Shumin Deng

**Abstract**: Recent progress in Multimodal Large Language Models (MLLMs) has unlocked powerful cross-modal reasoning abilities, but also raised new safety concerns, particularly when faced with adversarial multimodal inputs. To improve the safety of MLLMs during inference, we introduce a modular and adaptive inference-time intervention technology, AutoSteer, without requiring any fine-tuning of the underlying model. AutoSteer incorporates three core components: (1) a novel Safety Awareness Score (SAS) that automatically identifies the most safety-relevant distinctions among the model's internal layers; (2) an adaptive safety prober trained to estimate the likelihood of toxic outputs from intermediate representations; and (3) a lightweight Refusal Head that selectively intervenes to modulate generation when safety risks are detected. Experiments on LLaVA-OV and Chameleon across diverse safety-critical benchmarks demonstrate that AutoSteer significantly reduces the Attack Success Rate (ASR) for textual, visual, and cross-modal threats, while maintaining general abilities. These findings position AutoSteer as a practical, interpretable, and effective framework for safer deployment of multimodal AI systems.



## **30. Detecting LLM-generated Code with Subtle Modification by Adversarial Training**

cs.SE

**SubmitDate**: 2025-07-17    [abs](http://arxiv.org/abs/2507.13123v1) [paper-pdf](http://arxiv.org/pdf/2507.13123v1)

**Authors**: Xin Yin, Xinrui Li, Chao Ni, Xiaodan Xu, Xiaohu Yang

**Abstract**: With the rapid development of Large Language Models (LLMs), their powerful code-generation capabilities have been widely applied in tasks like code completion and automated development, demonstrating the value of improving coding efficiency. However, the extensive use of LLM-generated code also raises several new challenges. On the one hand, issues such as the regulation of code provenance, copyright disputes, and code quality have become increasingly concerning. How to effectively detect LLM-generated code and ensure its compliant and responsible use has become a critical and urgent issue. On the other hand, in practical applications, LLM-generated code is often subject to manual modifications, such as variable renaming or structural adjustments. Although some recent studies have proposed training-based and zero-shot methods for detecting LLM-generated code, these approaches show insufficient robustness when facing modified LLM-generated code, and there is a lack of an effective solution. To address the real-world scenario where LLM-generated code may undergo minor modifications, we propose CodeGPTSensor+, an enhanced version of CodeGPTSensor, which employs adversarial training to improve robustness against input perturbations. CodeGPTSensor+ integrates an adversarial sample generation module, Multi-objective Identifier and Structure Transformation (MIST), which systematically generates both high-quality and representative adversarial samples. This module effectively enhances the model's resistance against diverse adversarial attacks. Experimental results on the HMCorp dataset demonstrate that CodeGPTSensor+ significantly improves detection accuracy on the adversarial test set while maintaining high accuracy on the original test set, showcasing superior robustness compared to CodeGPTSensor.



## **31. MAD-Spear: A Conformity-Driven Prompt Injection Attack on Multi-Agent Debate Systems**

cs.CR

**SubmitDate**: 2025-07-17    [abs](http://arxiv.org/abs/2507.13038v1) [paper-pdf](http://arxiv.org/pdf/2507.13038v1)

**Authors**: Yu Cui, Hongyang Du

**Abstract**: Multi-agent debate (MAD) systems leverage collaborative interactions among large language models (LLMs) agents to improve reasoning capabilities. While recent studies have focused on increasing the accuracy and scalability of MAD systems, their security vulnerabilities have received limited attention. In this work, we introduce MAD-Spear, a targeted prompt injection attack that compromises a small subset of agents but significantly disrupts the overall MAD process. Manipulated agents produce multiple plausible yet incorrect responses, exploiting LLMs' conformity tendencies to propagate misinformation and degrade consensus quality. Furthermore, the attack can be composed with other strategies, such as communication attacks, to further amplify its impact by increasing the exposure of agents to incorrect responses. To assess MAD's resilience under attack, we propose a formal definition of MAD fault-tolerance and develop a comprehensive evaluation framework that jointly considers accuracy, consensus efficiency, and scalability. Extensive experiments on five benchmark datasets with varying difficulty levels demonstrate that MAD-Spear consistently outperforms the baseline attack in degrading system performance. Additionally, we observe that agent diversity substantially improves MAD performance in mathematical reasoning tasks, which challenges prior work suggesting that agent diversity has minimal impact on performance. These findings highlight the urgent need to improve the security in MAD design.



## **32. JailDAM: Jailbreak Detection with Adaptive Memory for Vision-Language Model**

cs.CR

**SubmitDate**: 2025-07-16    [abs](http://arxiv.org/abs/2504.03770v3) [paper-pdf](http://arxiv.org/pdf/2504.03770v3)

**Authors**: Yi Nian, Shenzhe Zhu, Yuehan Qin, Li Li, Ziyi Wang, Chaowei Xiao, Yue Zhao

**Abstract**: Multimodal large language models (MLLMs) excel in vision-language tasks but also pose significant risks of generating harmful content, particularly through jailbreak attacks. Jailbreak attacks refer to intentional manipulations that bypass safety mechanisms in models, leading to the generation of inappropriate or unsafe content. Detecting such attacks is critical to ensuring the responsible deployment of MLLMs. Existing jailbreak detection methods face three primary challenges: (1) Many rely on model hidden states or gradients, limiting their applicability to white-box models, where the internal workings of the model are accessible; (2) They involve high computational overhead from uncertainty-based analysis, which limits real-time detection, and (3) They require fully labeled harmful datasets, which are often scarce in real-world settings. To address these issues, we introduce a test-time adaptive framework called JAILDAM. Our method leverages a memory-based approach guided by policy-driven unsafe knowledge representations, eliminating the need for explicit exposure to harmful data. By dynamically updating unsafe knowledge during test-time, our framework improves generalization to unseen jailbreak strategies while maintaining efficiency. Experiments on multiple VLM jailbreak benchmarks demonstrate that JAILDAM delivers state-of-the-art performance in harmful content detection, improving both accuracy and speed.



## **33. SoK: Semantic Privacy in Large Language Models**

cs.CR

**SubmitDate**: 2025-07-16    [abs](http://arxiv.org/abs/2506.23603v2) [paper-pdf](http://arxiv.org/pdf/2506.23603v2)

**Authors**: Baihe Ma, Yanna Jiang, Xu Wang, Guangsheng Yu, Qin Wang, Caijun Sun, Chen Li, Xuelei Qi, Ying He, Wei Ni, Ren Ping Liu

**Abstract**: As Large Language Models (LLMs) are increasingly deployed in sensitive domains, traditional data privacy measures prove inadequate for protecting information that is implicit, contextual, or inferable - what we define as semantic privacy. This Systematization of Knowledge (SoK) introduces a lifecycle-centric framework to analyze how semantic privacy risks emerge across input processing, pretraining, fine-tuning, and alignment stages of LLMs. We categorize key attack vectors and assess how current defenses, such as differential privacy, embedding encryption, edge computing, and unlearning, address these threats. Our analysis reveals critical gaps in semantic-level protection, especially against contextual inference and latent representation leakage. We conclude by outlining open challenges, including quantifying semantic leakage, protecting multimodal inputs, balancing de-identification with generation quality, and ensuring transparency in privacy enforcement. This work aims to inform future research on designing robust, semantically aware privacy-preserving techniques for LLMs.



## **34. Thought Purity: Defense Paradigm For Chain-of-Thought Attack**

cs.LG

**SubmitDate**: 2025-07-16    [abs](http://arxiv.org/abs/2507.12314v1) [paper-pdf](http://arxiv.org/pdf/2507.12314v1)

**Authors**: Zihao Xue, Zhen Bi, Long Ma, Zhenlin Hu, Yan Wang, Zhenfang Liu, Qing Sheng, Jie Xiao, Jungang Lou

**Abstract**: While reinforcement learning-trained Large Reasoning Models (LRMs, e.g., Deepseek-R1) demonstrate advanced reasoning capabilities in the evolving Large Language Models (LLMs) domain, their susceptibility to security threats remains a critical vulnerability. This weakness is particularly evident in Chain-of-Thought (CoT) generation processes, where adversarial methods like backdoor prompt attacks can systematically subvert the model's core reasoning mechanisms. The emerging Chain-of-Thought Attack (CoTA) reveals this vulnerability through exploiting prompt controllability, simultaneously degrading both CoT safety and task performance with low-cost interventions. To address this compounded security-performance vulnerability, we propose Thought Purity (TP): a defense paradigm that systematically strengthens resistance to malicious content while preserving operational efficacy. Our solution achieves this through three synergistic components: (1) a safety-optimized data processing pipeline (2) reinforcement learning-enhanced rule constraints (3) adaptive monitoring metrics. Our approach establishes the first comprehensive defense mechanism against CoTA vulnerabilities in reinforcement learning-aligned reasoning systems, significantly advancing the security-functionality equilibrium for next-generation AI architectures.



## **35. Watch, Listen, Understand, Mislead: Tri-modal Adversarial Attacks on Short Videos for Content Appropriateness Evaluation**

cs.CV

Accepted as long paper, SVU Workshop at ICCV 2025

**SubmitDate**: 2025-07-16    [abs](http://arxiv.org/abs/2507.11968v1) [paper-pdf](http://arxiv.org/pdf/2507.11968v1)

**Authors**: Sahid Hossain Mustakim, S M Jishanul Islam, Ummay Maria Muna, Montasir Chowdhury, Mohammed Jawwadul Islam, Sadia Ahmmed, Tashfia Sikder, Syed Tasdid Azam Dhrubo, Swakkhar Shatabda

**Abstract**: Multimodal Large Language Models (MLLMs) are increasingly used for content moderation, yet their robustness in short-form video contexts remains underexplored. Current safety evaluations often rely on unimodal attacks, failing to address combined attack vulnerabilities. In this paper, we introduce a comprehensive framework for evaluating the tri-modal safety of MLLMs. First, we present the Short-Video Multimodal Adversarial (SVMA) dataset, comprising diverse short-form videos with human-guided synthetic adversarial attacks. Second, we propose ChimeraBreak, a novel tri-modal attack strategy that simultaneously challenges visual, auditory, and semantic reasoning pathways. Extensive experiments on state-of-the-art MLLMs reveal significant vulnerabilities with high Attack Success Rates (ASR). Our findings uncover distinct failure modes, showing model biases toward misclassifying benign or policy-violating content. We assess results using LLM-as-a-judge, demonstrating attack reasoning efficacy. Our dataset and findings provide crucial insights for developing more robust and safe MLLMs.



## **36. Seven Security Challenges That Must be Solved in Cross-domain Multi-agent LLM Systems**

cs.CR

**SubmitDate**: 2025-07-15    [abs](http://arxiv.org/abs/2505.23847v3) [paper-pdf](http://arxiv.org/pdf/2505.23847v3)

**Authors**: Ronny Ko, Jiseong Jeong, Shuyuan Zheng, Chuan Xiao, Tae-Wan Kim, Makoto Onizuka, Won-Yong Shin

**Abstract**: Large language models (LLMs) are rapidly evolving into autonomous agents that cooperate across organizational boundaries, enabling joint disaster response, supply-chain optimization, and other tasks that demand decentralized expertise without surrendering data ownership. Yet, cross-domain collaboration shatters the unified trust assumptions behind current alignment and containment techniques. An agent benign in isolation may, when receiving messages from an untrusted peer, leak secrets or violate policy, producing risks driven by emergent multi-agent dynamics rather than classical software bugs. This position paper maps the security agenda for cross-domain multi-agent LLM systems. We introduce seven categories of novel security challenges, for each of which we also present plausible attacks, security evaluation metrics, and future research guidelines.



## **37. A Generative Approach to LLM Harmfulness Detection with Special Red Flag Tokens**

cs.CL

14 pages, 6 figures

**SubmitDate**: 2025-07-15    [abs](http://arxiv.org/abs/2502.16366v3) [paper-pdf](http://arxiv.org/pdf/2502.16366v3)

**Authors**: Sophie Xhonneux, David Dobre, Mehrnaz Mofakhami, Leo Schwinn, Gauthier Gidel

**Abstract**: Most safety training methods for large language models (LLMs) are based on fine-tuning that forces models to shift from an unsafe answer to refusal when faced with harmful requests. Unfortunately, these drastic distribution shifts generally compromise model capabilities. To avoid that, we propose to expand the model's vocabulary with a special token we call red flag token (<rf>) and propose to train the model to insert this token into its response at any time when harmful content is generated or about to be generated. Our approach offers several advantages: it enables the model to explicitly learn the concept of harmfulness while marginally affecting the generated distribution, thus maintaining the model's utility. It also evaluates each generated answer and provides robustness as good as adversarial training without the need to run attacks during training. Moreover, by encapsulating our safety tuning in a LoRA module, we provide additional defenses against fine-tuning API attacks.



## **38. GUARD:Dual-Agent based Backdoor Defense on Chain-of-Thought in Neural Code Generation**

cs.SE

Accepted by SEKE 2025

**SubmitDate**: 2025-07-15    [abs](http://arxiv.org/abs/2505.21425v2) [paper-pdf](http://arxiv.org/pdf/2505.21425v2)

**Authors**: Naizhu Jin, Zhong Li, Tian Zhang, Qingkai Zeng

**Abstract**: With the widespread application of large language models in code generation, recent studies demonstrate that employing additional Chain-of-Thought generation models can significantly enhance code generation performance by providing explicit reasoning steps. However, as external components, CoT models are particularly vulnerable to backdoor attacks, which existing defense mechanisms often fail to detect effectively. To address this challenge, we propose GUARD, a novel dual-agent defense framework specifically designed to counter CoT backdoor attacks in neural code generation. GUARD integrates two core components: GUARD-Judge, which identifies suspicious CoT steps and potential triggers through comprehensive analysis, and GUARD-Repair, which employs a retrieval-augmented generation approach to regenerate secure CoT steps for identified anomalies. Experimental results show that GUARD effectively mitigates attacks while maintaining generation quality, advancing secure code generation systems.



## **39. Multi-Trigger Poisoning Amplifies Backdoor Vulnerabilities in LLMs**

cs.CL

**SubmitDate**: 2025-07-15    [abs](http://arxiv.org/abs/2507.11112v1) [paper-pdf](http://arxiv.org/pdf/2507.11112v1)

**Authors**: Sanhanat Sivapiromrat, Caiqi Zhang, Marco Basaldella, Nigel Collier

**Abstract**: Recent studies have shown that Large Language Models (LLMs) are vulnerable to data poisoning attacks, where malicious training examples embed hidden behaviours triggered by specific input patterns. However, most existing works assume a phrase and focus on the attack's effectiveness, offering limited understanding of trigger mechanisms and how multiple triggers interact within the model. In this paper, we present a framework for studying poisoning in LLMs. We show that multiple distinct backdoor triggers can coexist within a single model without interfering with each other, enabling adversaries to embed several triggers concurrently. Using multiple triggers with high embedding similarity, we demonstrate that poisoned triggers can achieve robust activation even when tokens are substituted or separated by long token spans. Our findings expose a broader and more persistent vulnerability surface in LLMs. To mitigate this threat, we propose a post hoc recovery method that selectively retrains specific model components based on a layer-wise weight difference analysis. Our method effectively removes the trigger behaviour with minimal parameter updates, presenting a practical and efficient defence against multi-trigger poisoning.



## **40. The Devil behind the mask: An emergent safety vulnerability of Diffusion LLMs**

cs.CL

21 pages, 9 figures, work in progress

**SubmitDate**: 2025-07-15    [abs](http://arxiv.org/abs/2507.11097v1) [paper-pdf](http://arxiv.org/pdf/2507.11097v1)

**Authors**: Zichen Wen, Jiashu Qu, Dongrui Liu, Zhiyuan Liu, Ruixi Wu, Yicun Yang, Xiangqi Jin, Haoyun Xu, Xuyang Liu, Weijia Li, Chaochao Lu, Jing Shao, Conghui He, Linfeng Zhang

**Abstract**: Diffusion-based large language models (dLLMs) have recently emerged as a powerful alternative to autoregressive LLMs, offering faster inference and greater interactivity via parallel decoding and bidirectional modeling. However, despite strong performance in code generation and text infilling, we identify a fundamental safety concern: existing alignment mechanisms fail to safeguard dLLMs against context-aware, masked-input adversarial prompts, exposing novel vulnerabilities. To this end, we present DIJA, the first systematic study and jailbreak attack framework that exploits unique safety weaknesses of dLLMs. Specifically, our proposed DIJA constructs adversarial interleaved mask-text prompts that exploit the text generation mechanisms of dLLMs, i.e., bidirectional modeling and parallel decoding. Bidirectional modeling drives the model to produce contextually consistent outputs for masked spans, even when harmful, while parallel decoding limits model dynamic filtering and rejection sampling of unsafe content. This causes standard alignment mechanisms to fail, enabling harmful completions in alignment-tuned dLLMs, even when harmful behaviors or unsafe instructions are directly exposed in the prompt. Through comprehensive experiments, we demonstrate that DIJA significantly outperforms existing jailbreak methods, exposing a previously overlooked threat surface in dLLM architectures. Notably, our method achieves up to 100% keyword-based ASR on Dream-Instruct, surpassing the strongest prior baseline, ReNeLLM, by up to 78.5% in evaluator-based ASR on JailbreakBench and by 37.7 points in StrongREJECT score, while requiring no rewriting or hiding of harmful content in the jailbreak prompt. Our findings underscore the urgent need for rethinking safety alignment in this emerging class of language models. Code is available at https://github.com/ZichenWen1/DIJA.



## **41. Representation Bending for Large Language Model Safety**

cs.LG

Accepted to ACL 2025 (main)

**SubmitDate**: 2025-07-15    [abs](http://arxiv.org/abs/2504.01550v3) [paper-pdf](http://arxiv.org/pdf/2504.01550v3)

**Authors**: Ashkan Yousefpour, Taeheon Kim, Ryan S. Kwon, Seungbeen Lee, Wonje Jeung, Seungju Han, Alvin Wan, Harrison Ngan, Youngjae Yu, Jonghyun Choi

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools, but their inherent safety risks - ranging from harmful content generation to broader societal harms - pose significant challenges. These risks can be amplified by the recent adversarial attacks, fine-tuning vulnerabilities, and the increasing deployment of LLMs in high-stakes environments. Existing safety-enhancing techniques, such as fine-tuning with human feedback or adversarial training, are still vulnerable as they address specific threats and often fail to generalize across unseen attacks, or require manual system-level defenses. This paper introduces RepBend, a novel approach that fundamentally disrupts the representations underlying harmful behaviors in LLMs, offering a scalable solution to enhance (potentially inherent) safety. RepBend brings the idea of activation steering - simple vector arithmetic for steering model's behavior during inference - to loss-based fine-tuning. Through extensive evaluation, RepBend achieves state-of-the-art performance, outperforming prior methods such as Circuit Breaker, RMU, and NPO, with up to 95% reduction in attack success rates across diverse jailbreak benchmarks, all with negligible reduction in model usability and general capabilities.



## **42. From Alerts to Intelligence: A Novel LLM-Aided Framework for Host-based Intrusion Detection**

cs.CR

**SubmitDate**: 2025-07-15    [abs](http://arxiv.org/abs/2507.10873v1) [paper-pdf](http://arxiv.org/pdf/2507.10873v1)

**Authors**: Danyu Sun, Jinghuai Zhang, Jiacen Xu, Yu Zheng, Yuan Tian, Zhou Li

**Abstract**: Host-based intrusion detection system (HIDS) is a key defense component to protect the organizations from advanced threats like Advanced Persistent Threats (APT). By analyzing the fine-grained logs with approaches like data provenance, HIDS has shown successes in capturing sophisticated attack traces. Despite the progresses embarked by the research community and industry, HIDS still frequently encounters backlash from their operators in the deployed environments, due to issues like high false-positive rate, inconsistent outcomes across environments and human-unfriendly detection results. Large Language Models (LLMs) have great potentials to advance the state of HIDS, given their extensive knowledge of attack techniques and their ability to detect anomalies through semantic analysis, anchored by recent studies. Yet, our preliminary analysis indicates that building an HIDS by naively prompting an LLM is unlikely to succeed. In this work, we explore the direction of building a customized LLM pipeline for HIDS and develop a system named SHIELD. SHIELD addresses challenges related to LLM's token limits, confusion of background noises, etc., by integrating a variety of techniques like event-level Masked Autoencoder (MAE) for attack window detection, attack evidence identification and expansion, Deterministic Data Augmentation (DDA) for profiling normal activities, and multi-purpose prompting that guides the LLM to conduct precise and interpretable attack investigations. Extensive experiments on three log datasets (DARPA-E3, NodLink-simulated-data and ATLASv2) show that SHIELD consistently achieves outstanding performance in comparison with 5 representative HIDS. These findings highlight the potential of LLMs as powerful tools for intrusion detection and pave the way for future research in this domain.



## **43. REAL-IoT: Characterizing GNN Intrusion Detection Robustness under Practical Adversarial Attack**

cs.CR

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2507.10836v1) [paper-pdf](http://arxiv.org/pdf/2507.10836v1)

**Authors**: Zhonghao Zhan, Huichi Zhou, Hamed Haddadi

**Abstract**: Graph Neural Network (GNN)-based network intrusion detection systems (NIDS) are often evaluated on single datasets, limiting their ability to generalize under distribution drift. Furthermore, their adversarial robustness is typically assessed using synthetic perturbations that lack realism. This measurement gap leads to an overestimation of GNN-based NIDS resilience. To address the limitations, we propose \textbf{REAL-IoT}, a comprehensive framework for robustness evaluation of GNN-based NIDS in IoT environments. Our framework presents a methodology that creates a unified dataset from canonical datasets to assess generalization under drift. In addition, it features a novel intrusion dataset collected from a physical IoT testbed, which captures network traffic and attack scenarios under real-world settings. Furthermore, using REAL-IoT, we explore the usage of Large Language Models (LLMs) to analyze network data and mitigate the impact of adversarial examples by filtering suspicious flows. Our evaluations using REAL-IoT reveal performance drops in GNN models compared to results from standard benchmarks, quantifying their susceptibility to drift and realistic attacks. We also demonstrate the potential of LLM-based filtering to enhance robustness. These findings emphasize the necessity of realistic threat modeling and rigorous measurement practices for developing resilient IoT intrusion detection systems.



## **44. Model Tampering Attacks Enable More Rigorous Evaluations of LLM Capabilities**

cs.CR

Accepted to TMLR

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2502.05209v3) [paper-pdf](http://arxiv.org/pdf/2502.05209v3)

**Authors**: Zora Che, Stephen Casper, Robert Kirk, Anirudh Satheesh, Stewart Slocum, Lev E McKinney, Rohit Gandikota, Aidan Ewart, Domenic Rosati, Zichu Wu, Zikui Cai, Bilal Chughtai, Yarin Gal, Furong Huang, Dylan Hadfield-Menell

**Abstract**: Evaluations of large language model (LLM) risks and capabilities are increasingly being incorporated into AI risk management and governance frameworks. Currently, most risk evaluations are conducted by designing inputs that elicit harmful behaviors from the system. However, this approach suffers from two limitations. First, input-output evaluations cannot fully evaluate realistic risks from open-weight models. Second, the behaviors identified during any particular input-output evaluation can only lower-bound the model's worst-possible-case input-output behavior. As a complementary method for eliciting harmful behaviors, we propose evaluating LLMs with model tampering attacks which allow for modifications to latent activations or weights. We pit state-of-the-art techniques for removing harmful LLM capabilities against a suite of 5 input-space and 6 model tampering attacks. In addition to benchmarking these methods against each other, we show that (1) model resilience to capability elicitation attacks lies on a low-dimensional robustness subspace; (2) the success rate of model tampering attacks can empirically predict and offer conservative estimates for the success of held-out input-space attacks; and (3) state-of-the-art unlearning methods can easily be undone within 16 steps of fine-tuning. Together, these results highlight the difficulty of suppressing harmful LLM capabilities and show that model tampering attacks enable substantially more rigorous evaluations than input-space attacks alone.



## **45. PRM-Free Security Alignment of Large Models via Red Teaming and Adversarial Training**

cs.CR

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2507.14202v1) [paper-pdf](http://arxiv.org/pdf/2507.14202v1)

**Authors**: Pengfei Du

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across diverse applications, yet they pose significant security risks that threaten their safe deployment in critical domains. Current security alignment methodologies predominantly rely on Process Reward Models (PRMs) to evaluate intermediate reasoning steps, introducing substantial computational overhead and scalability constraints. This paper presents a novel PRM-free security alignment framework that leverages automated red teaming and adversarial training to achieve robust security guarantees while maintaining computational efficiency. Our approach systematically identifies vulnerabilities through sophisticated attack strategies including genetic algorithm optimization, multi-agent simulation, and advanced prompt mutation techniques. The framework enhances model robustness via targeted adversarial training with curriculum learning and adaptive regularization mechanisms. Comprehensive experimental evaluation across five state-of-the-art LLMs demonstrates that our method achieves superior security alignment performance compared to PRM-based approaches while reducing computational costs by 61\%. The framework incorporates transparent reporting and continuous audit mechanisms that enable iterative security improvement and regulatory compliance. Our contributions advance the field of efficient LLM security alignment by democratizing access to robust security measures for resource-constrained organizations and providing a scalable foundation for addressing evolving adversarial threats.



## **46. Logic layer Prompt Control Injection (LPCI): A Novel Security Vulnerability Class in Agentic Systems**

cs.CR

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2507.10457v1) [paper-pdf](http://arxiv.org/pdf/2507.10457v1)

**Authors**: Hammad Atta, Ken Huang, Manish Bhatt, Kamal Ahmed, Muhammad Aziz Ul Haq, Yasir Mehmood

**Abstract**: The integration of large language models (LLMs) into enterprise systems has created a new class of covert security vulnerabilities, particularly within logic-execution layers and persistent-memory contexts. In this paper, we introduce Logic-Layer Prompt Control Injection (LPCI), a novel attack category in which encoded, delayed, and conditionally triggered payloads are embedded in memory, vector stores, or tool outputs. These payloads can bypass conventional input filters and trigger unauthorised behaviour across sessions.



## **47. Bypassing LLM Guardrails: An Empirical Analysis of Evasion Attacks against Prompt Injection and Jailbreak Detection Systems**

cs.CR

14 pages, 5 figures, 11 tables. To be published in LLMSec 2025

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2504.11168v3) [paper-pdf](http://arxiv.org/pdf/2504.11168v3)

**Authors**: William Hackett, Lewis Birch, Stefan Trawicki, Neeraj Suri, Peter Garraghan

**Abstract**: Large Language Models (LLMs) guardrail systems are designed to protect against prompt injection and jailbreak attacks. However, they remain vulnerable to evasion techniques. We demonstrate two approaches for bypassing LLM prompt injection and jailbreak detection systems via traditional character injection methods and algorithmic Adversarial Machine Learning (AML) evasion techniques. Through testing against six prominent protection systems, including Microsoft's Azure Prompt Shield and Meta's Prompt Guard, we show that both methods can be used to evade detection while maintaining adversarial utility achieving in some instances up to 100% evasion success. Furthermore, we demonstrate that adversaries can enhance Attack Success Rates (ASR) against black-box targets by leveraging word importance ranking computed by offline white-box models. Our findings reveal vulnerabilities within current LLM protection mechanisms and highlight the need for more robust guardrail systems.



## **48. ARMOR: Aligning Secure and Safe Large Language Models via Meticulous Reasoning**

cs.CR

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2507.11500v1) [paper-pdf](http://arxiv.org/pdf/2507.11500v1)

**Authors**: Zhengyue Zhao, Yingzi Ma, Somesh Jha, Marco Pavone, Chaowei Xiao

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable generative capabilities. However, their susceptibility to misuse has raised significant safety concerns. While post-training safety alignment methods have been widely adopted, LLMs remain vulnerable to malicious instructions that can bypass safety constraints. Recent efforts have introduced inference-time safety reasoning (system-2 alignment), where LLMs conduct a reasoning process to perform safety verification before final response. We show, however, that these checks are driven by ad-hoc reasoning that diverges from the structured human process, where they first discern a user's true intent, then evaluate the associated risk based on the true intent. Consequently, these defenses remain vulnerable to sophisticated jailbreak prompts that cloak harmful goals in seemingly benign language. To build secure and safe LLMs, we propose a reasoning-based safety alignment framework, ARMOR, that replaces the ad-hoc chains of thought reasoning process with human-aligned, structured one. At inference, ARMOR (1) detects likely jailbreak strategies, (2) extracts the user's core intent while discarding deceptive instructions, and (3) applies a policy-grounded safety analysis to the purified request. ARMOR is evaluated on adaptive jailbreak attacks and multiple safety benchmarks, and a test-time scaling is conducted to further improve its performance. Results demonstrate that ARMOR significantly enhances the robustness against state-of-the-art adaptive jailbreak attacks and outperforms recent reasoning-based aligned models across various safety benchmarks.



## **49. IPAD: Inverse Prompt for AI Detection -- A Robust and Explainable LLM-Generated Text Detector**

cs.LG

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2502.15902v2) [paper-pdf](http://arxiv.org/pdf/2502.15902v2)

**Authors**: Zheng Chen, Yushi Feng, Changyang He, Yue Deng, Hongxi Pu, Bo Li

**Abstract**: Large Language Models (LLMs) have attained human-level fluency in text generation, which complicates the distinction between human-written and LLM-generated texts. This increases the risk of misuse and highlights the need for reliable detectors. Yet, existing detectors exhibit poor robustness on out-of-distribution (OOD) data and attacked data, which is critical for real-world scenarios. Also, they struggle to provide interpretable evidence to support their decisions, thus undermining the reliability. In light of these challenges, we propose IPAD (Inverse Prompt for AI Detection), a novel framework consisting of a Prompt Inverter that identifies predicted prompts that could have generated the input text, and two Distinguishers that examine the probability that the input texts align with the predicted prompts. Empirical evaluations demonstrate that IPAD outperforms the strongest baselines by 9.05% (Average Recall) on in-distribution data, 12.93% (AUROC) on out-of-distribution (OOD) data, and 5.48% (AUROC) on attacked data. IPAD also performs robustly on structured datasets. Furthermore, an interpretability assessment is conducted to illustrate that IPAD enhances the AI detection trustworthiness by allowing users to directly examine the decision-making evidence, which provides interpretable support for its state-of-the-art detection results.



## **50. The Man Behind the Sound: Demystifying Audio Private Attribute Profiling via Multimodal Large Language Model Agents**

cs.CR

22 pages, 4 figures

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2507.10016v1) [paper-pdf](http://arxiv.org/pdf/2507.10016v1)

**Authors**: Lixu Wang, Kaixiang Yao, Xinfeng Li, Dong Yang, Haoyang Li, Xiaofeng Wang, Wei Dong

**Abstract**: Our research uncovers a novel privacy risk associated with multimodal large language models (MLLMs): the ability to infer sensitive personal attributes from audio data -- a technique we term audio private attribute profiling. This capability poses a significant threat, as audio can be covertly captured without direct interaction or visibility. Moreover, compared to images and text, audio carries unique characteristics, such as tone and pitch, which can be exploited for more detailed profiling. However, two key challenges exist in understanding MLLM-employed private attribute profiling from audio: (1) the lack of audio benchmark datasets with sensitive attribute annotations and (2) the limited ability of current MLLMs to infer such attributes directly from audio. To address these challenges, we introduce AP^2, an audio benchmark dataset that consists of two subsets collected and composed from real-world data, and both are annotated with sensitive attribute labels. Additionally, we propose Gifts, a hybrid multi-agent framework that leverages the complementary strengths of audio-language models (ALMs) and large language models (LLMs) to enhance inference capabilities. Gifts employs an LLM to guide the ALM in inferring sensitive attributes, then forensically analyzes and consolidates the ALM's inferences, overcoming severe hallucinations of existing ALMs in generating long-context responses. Our evaluations demonstrate that Gifts significantly outperforms baseline approaches in inferring sensitive attributes. Finally, we investigate model-level and data-level defense strategies to mitigate the risks of audio private attribute profiling. Our work validates the feasibility of audio-based privacy attacks using MLLMs, highlighting the need for robust defenses, and provides a dataset and framework to facilitate future research.



