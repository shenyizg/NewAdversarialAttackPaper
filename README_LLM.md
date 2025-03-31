# Latest Large Language Model Attack Papers
**update at 2025-03-31 15:45:57**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. Training Large Language Models for Advanced Typosquatting Detection**

cs.CR

6 pages, 1 figure

**SubmitDate**: 2025-03-28    [abs](http://arxiv.org/abs/2503.22406v1) [paper-pdf](http://arxiv.org/pdf/2503.22406v1)

**Authors**: Jackson Welch

**Abstract**: Typosquatting is a long-standing cyber threat that exploits human error in typing URLs to deceive users, distribute malware, and conduct phishing attacks. With the proliferation of domain names and new Top-Level Domains (TLDs), typosquatting techniques have grown more sophisticated, posing significant risks to individuals, businesses, and national cybersecurity infrastructure. Traditional detection methods primarily focus on well-known impersonation patterns, leaving gaps in identifying more complex attacks. This study introduces a novel approach leveraging large language models (LLMs) to enhance typosquatting detection. By training an LLM on character-level transformations and pattern-based heuristics rather than domain-specific data, a more adaptable and resilient detection mechanism develops. Experimental results indicate that the Phi-4 14B model outperformed other tested models when properly fine tuned achieving a 98% accuracy rate with only a few thousand training samples. This research highlights the potential of LLMs in cybersecurity applications, specifically in mitigating domain-based deception tactics, and provides insights into optimizing machine learning strategies for threat detection.



## **2. Single Image Unlearning: Efficient Machine Unlearning in Multimodal Large Language Models**

cs.CV

**SubmitDate**: 2025-03-28    [abs](http://arxiv.org/abs/2405.12523v3) [paper-pdf](http://arxiv.org/pdf/2405.12523v3)

**Authors**: Jiaqi Li, Qianshan Wei, Chuanyi Zhang, Guilin Qi, Miaozeng Du, Yongrui Chen, Sheng Bi, Fan Liu

**Abstract**: Machine unlearning empowers individuals with the `right to be forgotten' by removing their private or sensitive information encoded in machine learning models. However, it remains uncertain whether MU can be effectively applied to Multimodal Large Language Models (MLLMs), particularly in scenarios of forgetting the leaked visual data of concepts. To overcome the challenge, we propose an efficient method, Single Image Unlearning (SIU), to unlearn the visual recognition of a concept by fine-tuning a single associated image for few steps. SIU consists of two key aspects: (i) Constructing Multifaceted fine-tuning data. We introduce four targets, based on which we construct fine-tuning data for the concepts to be forgotten; (ii) Jointly training loss. To synchronously forget the visual recognition of concepts and preserve the utility of MLLMs, we fine-tune MLLMs through a novel Dual Masked KL-divergence Loss combined with Cross Entropy loss. Alongside our method, we establish MMUBench, a new benchmark for MU in MLLMs and introduce a collection of metrics for its evaluation. Experimental results on MMUBench show that SIU completely surpasses the performance of existing methods. Furthermore, we surprisingly find that SIU can avoid invasive membership inference attacks and jailbreak attacks. To the best of our knowledge, we are the first to explore MU in MLLMs. We will release the code and benchmark in the near future.



## **3. AnyAttack: Towards Large-scale Self-supervised Adversarial Attacks on Vision-language Models**

cs.LG

CVPR 2025

**SubmitDate**: 2025-03-28    [abs](http://arxiv.org/abs/2410.05346v3) [paper-pdf](http://arxiv.org/pdf/2410.05346v3)

**Authors**: Jiaming Zhang, Junhong Ye, Xingjun Ma, Yige Li, Yunfan Yang, Yunhao Chen, Jitao Sang, Dit-Yan Yeung

**Abstract**: Due to their multimodal capabilities, Vision-Language Models (VLMs) have found numerous impactful applications in real-world scenarios. However, recent studies have revealed that VLMs are vulnerable to image-based adversarial attacks. Traditional targeted adversarial attacks require specific targets and labels, limiting their real-world impact.We present AnyAttack, a self-supervised framework that transcends the limitations of conventional attacks through a novel foundation model approach. By pre-training on the massive LAION-400M dataset without label supervision, AnyAttack achieves unprecedented flexibility - enabling any image to be transformed into an attack vector targeting any desired output across different VLMs.This approach fundamentally changes the threat landscape, making adversarial capabilities accessible at an unprecedented scale. Our extensive validation across five open-source VLMs (CLIP, BLIP, BLIP2, InstructBLIP, and MiniGPT-4) demonstrates AnyAttack's effectiveness across diverse multimodal tasks. Most concerning, AnyAttack seamlessly transfers to commercial systems including Google Gemini, Claude Sonnet, Microsoft Copilot and OpenAI GPT, revealing a systemic vulnerability requiring immediate attention.



## **4. Foot-In-The-Door: A Multi-turn Jailbreak for LLMs**

cs.CL

19 pages, 8 figures

**SubmitDate**: 2025-03-28    [abs](http://arxiv.org/abs/2502.19820v3) [paper-pdf](http://arxiv.org/pdf/2502.19820v3)

**Authors**: Zixuan Weng, Xiaolong Jin, Jinyuan Jia, Xiangyu Zhang

**Abstract**: Ensuring AI safety is crucial as large language models become increasingly integrated into real-world applications. A key challenge is jailbreak, where adversarial prompts bypass built-in safeguards to elicit harmful disallowed outputs. Inspired by psychological foot-in-the-door principles, we introduce FITD,a novel multi-turn jailbreak method that leverages the phenomenon where minor initial commitments lower resistance to more significant or more unethical transgressions. Our approach progressively escalates the malicious intent of user queries through intermediate bridge prompts and aligns the model's response by itself to induce toxic responses. Extensive experimental results on two jailbreak benchmarks demonstrate that FITD achieves an average attack success rate of 94% across seven widely used models, outperforming existing state-of-the-art methods. Additionally, we provide an in-depth analysis of LLM self-corruption, highlighting vulnerabilities in current alignment strategies and emphasizing the risks inherent in multi-turn interactions. The code is available at https://github.com/Jinxiaolong1129/Foot-in-the-door-Jailbreak.



## **5. Debate-Driven Multi-Agent LLMs for Phishing Email Detection**

cs.MA

Accepted to the 13th International Symposium on Digital Forensics and  Security (ISDFS 2025)

**SubmitDate**: 2025-03-27    [abs](http://arxiv.org/abs/2503.22038v1) [paper-pdf](http://arxiv.org/pdf/2503.22038v1)

**Authors**: Ngoc Tuong Vy Nguyen, Felix D Childress, Yunting Yin

**Abstract**: Phishing attacks remain a critical cybersecurity threat. Attackers constantly refine their methods, making phishing emails harder to detect. Traditional detection methods, including rule-based systems and supervised machine learning models, either rely on predefined patterns like blacklists, which can be bypassed with slight modifications, or require large datasets for training and still can generate false positives and false negatives. In this work, we propose a multi-agent large language model (LLM) prompting technique that simulates debates among agents to detect whether the content presented on an email is phishing. Our approach uses two LLM agents to present arguments for or against the classification task, with a judge agent adjudicating the final verdict based on the quality of reasoning provided. This debate mechanism enables the models to critically analyze contextual cue and deceptive patterns in text, which leads to improved classification accuracy. The proposed framework is evaluated on multiple phishing email datasets and demonstrate that mixed-agent configurations consistently outperform homogeneous configurations. Results also show that the debate structure itself is sufficient to yield accurate decisions without extra prompting strategies.



## **6. Intelligent IoT Attack Detection Design via ODLLM with Feature Ranking-based Knowledge Base**

cs.CR

**SubmitDate**: 2025-03-27    [abs](http://arxiv.org/abs/2503.21674v1) [paper-pdf](http://arxiv.org/pdf/2503.21674v1)

**Authors**: Satvik Verma, Qun Wang, E. Wes Bethel

**Abstract**: The widespread adoption of Internet of Things (IoT) devices has introduced significant cybersecurity challenges, particularly with the increasing frequency and sophistication of Distributed Denial of Service (DDoS) attacks. Traditional machine learning (ML) techniques often fall short in detecting such attacks due to the complexity of blended and evolving patterns. To address this, we propose a novel framework leveraging On-Device Large Language Models (ODLLMs) augmented with fine-tuning and knowledge base (KB) integration for intelligent IoT network attack detection. By implementing feature ranking techniques and constructing both long and short KBs tailored to model capacities, the proposed framework ensures efficient and accurate detection of DDoS attacks while overcoming computational and privacy limitations. Simulation results demonstrate that the optimized framework achieves superior accuracy across diverse attack types, especially when using compact models in edge computing environments. This work provides a scalable and secure solution for real-time IoT security, advancing the applicability of edge intelligence in cybersecurity.



## **7. CleanGen: Mitigating Backdoor Attacks for Generation Tasks in Large Language Models**

cs.AI

This paper is presented at EMNLP 2024

**SubmitDate**: 2025-03-27    [abs](http://arxiv.org/abs/2406.12257v3) [paper-pdf](http://arxiv.org/pdf/2406.12257v3)

**Authors**: Yuetai Li, Zhangchen Xu, Fengqing Jiang, Luyao Niu, Dinuka Sahabandu, Bhaskar Ramasubramanian, Radha Poovendran

**Abstract**: The remarkable performance of large language models (LLMs) in generation tasks has enabled practitioners to leverage publicly available models to power custom applications, such as chatbots and virtual assistants. However, the data used to train or fine-tune these LLMs is often undisclosed, allowing an attacker to compromise the data and inject backdoors into the models. In this paper, we develop a novel inference time defense, named CLEANGEN, to mitigate backdoor attacks for generation tasks in LLMs. CLEANGEN is a lightweight and effective decoding strategy that is compatible with the state-of-the-art (SOTA) LLMs. Our insight behind CLEANGEN is that compared to other LLMs, backdoored LLMs assign significantly higher probabilities to tokens representing the attacker-desired contents. These discrepancies in token probabilities enable CLEANGEN to identify suspicious tokens favored by the attacker and replace them with tokens generated by another LLM that is not compromised by the same attacker, thereby avoiding generation of attacker-desired content. We evaluate CLEANGEN against five SOTA backdoor attacks. Our results show that CLEANGEN achieves lower attack success rates (ASR) compared to five SOTA baseline defenses for all five backdoor attacks. Moreover, LLMs deploying CLEANGEN maintain helpfulness in their responses when serving benign user queries with minimal added computational overhead.



## **8. Harnessing Chain-of-Thought Metadata for Task Routing and Adversarial Prompt Detection**

cs.CL

**SubmitDate**: 2025-03-27    [abs](http://arxiv.org/abs/2503.21464v1) [paper-pdf](http://arxiv.org/pdf/2503.21464v1)

**Authors**: Ryan Marinelli, Josef Pichlmeier, Tamas Bisztray

**Abstract**: In this work, we propose a metric called Number of Thoughts (NofT) to determine the difficulty of tasks pre-prompting and support Large Language Models (LLMs) in production contexts. By setting thresholds based on the number of thoughts, this metric can discern the difficulty of prompts and support more effective prompt routing. A 2% decrease in latency is achieved when routing prompts from the MathInstruct dataset through quantized, distilled versions of Deepseek with 1.7 billion, 7 billion, and 14 billion parameters. Moreover, this metric can be used to detect adversarial prompts used in prompt injection attacks with high efficacy. The Number of Thoughts can inform a classifier that achieves 95% accuracy in adversarial prompt detection. Our experiments ad datasets used are available on our GitHub page: https://github.com/rymarinelli/Number_Of_Thoughts/tree/main.



## **9. Tricking Retrievers with Influential Tokens: An Efficient Black-Box Corpus Poisoning Attack**

cs.LG

Accepted to NAACL 2025 Main Track

**SubmitDate**: 2025-03-27    [abs](http://arxiv.org/abs/2503.21315v1) [paper-pdf](http://arxiv.org/pdf/2503.21315v1)

**Authors**: Cheng Wang, Yiwei Wang, Yujun Cai, Bryan Hooi

**Abstract**: Retrieval-augmented generation (RAG) systems enhance large language models by incorporating external knowledge, addressing issues like outdated internal knowledge and hallucination. However, their reliance on external knowledge bases makes them vulnerable to corpus poisoning attacks, where adversarial passages can be injected to manipulate retrieval results. Existing methods for crafting such passages, such as random token replacement or training inversion models, are often slow and computationally expensive, requiring either access to retriever's gradients or large computational resources. To address these limitations, we propose Dynamic Importance-Guided Genetic Algorithm (DIGA), an efficient black-box method that leverages two key properties of retrievers: insensitivity to token order and bias towards influential tokens. By focusing on these characteristics, DIGA dynamically adjusts its genetic operations to generate effective adversarial passages with significantly reduced time and memory usage. Our experimental evaluation shows that DIGA achieves superior efficiency and scalability compared to existing methods, while maintaining comparable or better attack success rates across multiple datasets.



## **10. M-LLM Based Video Frame Selection for Efficient Video Understanding**

cs.CV

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2502.19680v2) [paper-pdf](http://arxiv.org/pdf/2502.19680v2)

**Authors**: Kai Hu, Feng Gao, Xiaohan Nie, Peng Zhou, Son Tran, Tal Neiman, Lingyun Wang, Mubarak Shah, Raffay Hamid, Bing Yin, Trishul Chilimbi

**Abstract**: Recent advances in Multi-Modal Large Language Models (M-LLMs) show promising results in video reasoning. Popular Multi-Modal Large Language Model (M-LLM) frameworks usually apply naive uniform sampling to reduce the number of video frames that are fed into an M-LLM, particularly for long context videos. However, it could lose crucial context in certain periods of a video, so that the downstream M-LLM may not have sufficient visual information to answer a question. To attack this pain point, we propose a light-weight M-LLM -based frame selection method that adaptively select frames that are more relevant to users' queries. In order to train the proposed frame selector, we introduce two supervision signals (i) Spatial signal, where single frame importance score by prompting a M-LLM; (ii) Temporal signal, in which multiple frames selection by prompting Large Language Model (LLM) using the captions of all frame candidates. The selected frames are then digested by a frozen downstream video M-LLM for visual reasoning and question answering. Empirical results show that the proposed M-LLM video frame selector improves the performances various downstream video Large Language Model (video-LLM) across medium (ActivityNet, NExT-QA) and long (EgoSchema, LongVideoBench) context video question answering benchmarks.



## **11. Iterative Prompting with Persuasion Skills in Jailbreaking Large Language Models**

cs.CL

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2503.20320v1) [paper-pdf](http://arxiv.org/pdf/2503.20320v1)

**Authors**: Shih-Wen Ke, Guan-Yu Lai, Guo-Lin Fang, Hsi-Yuan Kao

**Abstract**: Large language models (LLMs) are designed to align with human values in their responses. This study exploits LLMs with an iterative prompting technique where each prompt is systematically modified and refined across multiple iterations to enhance its effectiveness in jailbreaking attacks progressively. This technique involves analyzing the response patterns of LLMs, including GPT-3.5, GPT-4, LLaMa2, Vicuna, and ChatGLM, allowing us to adjust and optimize prompts to evade the LLMs' ethical and security constraints. Persuasion strategies enhance prompt effectiveness while maintaining consistency with malicious intent. Our results show that the attack success rates (ASR) increase as the attacking prompts become more refined with the highest ASR of 90% for GPT4 and ChatGLM and the lowest ASR of 68% for LLaMa2. Our technique outperforms baseline techniques (PAIR and PAP) in ASR and shows comparable performance with GCG and ArtPrompt.



## **12. sudo rm -rf agentic_security**

cs.CL

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2503.20279v1) [paper-pdf](http://arxiv.org/pdf/2503.20279v1)

**Authors**: Sejin Lee, Jian Kim, Haon Park, Ashkan Yousefpour, Sangyoon Yu, Min Song

**Abstract**: Large Language Models (LLMs) are increasingly deployed as computer-use agents, autonomously performing tasks within real desktop or web environments. While this evolution greatly expands practical use cases for humans, it also creates serious security exposures. We present SUDO (Screen-based Universal Detox2Tox Offense), a novel attack framework that systematically bypasses refusal trained safeguards in commercial computer-use agents, such as Claude Computer Use. The core mechanism, Detox2Tox, transforms harmful requests (that agents initially reject) into seemingly benign requests via detoxification, secures detailed instructions from advanced vision language models (VLMs), and then reintroduces malicious content via toxification just before execution. Unlike conventional jailbreaks, SUDO iteratively refines its attacks based on a built-in refusal feedback, making it increasingly effective against robust policy filters. In extensive tests spanning 50 real-world tasks and multiple state-of-the-art VLMs, SUDO achieves a stark attack success rate of 24% (with no refinement), and up to 41% (by its iterative refinement) in Claude Computer Use. By revealing these vulnerabilities and demonstrating the ease with which they can be exploited in real-world computing environments, this paper highlights an immediate need for robust, context-aware safeguards. WARNING: This paper includes harmful or offensive model outputs.



## **13. Stealthy Backdoor Attack in Self-Supervised Learning Vision Encoders for Large Vision Language Models**

cs.CV

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2502.18290v3) [paper-pdf](http://arxiv.org/pdf/2502.18290v3)

**Authors**: Zhaoyi Liu, Huan Zhang

**Abstract**: Self-supervised learning (SSL) vision encoders learn high-quality image representations and thus have become a vital part of developing vision modality of large vision language models (LVLMs). Due to the high cost of training such encoders, pre-trained encoders are widely shared and deployed into many LVLMs, which are security-critical or bear societal significance. Under this practical scenario, we reveal a new backdoor threat that significant visual hallucinations can be induced into these LVLMs by merely compromising vision encoders. Because of the sharing and reuse of these encoders, many downstream LVLMs may inherit backdoor behaviors from encoders, leading to widespread backdoors. In this work, we propose BadVision, the first method to exploit this vulnerability in SSL vision encoders for LVLMs with novel trigger optimization and backdoor learning techniques. We evaluate BadVision on two types of SSL encoders and LVLMs across eight benchmarks. We show that BadVision effectively drives the LVLMs to attacker-chosen hallucination with over 99% attack success rate, causing a 77.6% relative visual understanding error while maintaining the stealthiness. SoTA backdoor detection methods cannot detect our attack effectively.



## **14. TeleLoRA: Teleporting Model-Specific Alignment Across LLMs**

cs.LG

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2503.20228v1) [paper-pdf](http://arxiv.org/pdf/2503.20228v1)

**Authors**: Xiao Lin, Manoj Acharya, Anirban Roy, Susmit Jha

**Abstract**: Mitigating Trojans in Large Language Models (LLMs) is one of many tasks where alignment data is LLM specific, as different LLMs have different Trojan triggers and trigger behaviors to be removed. In this paper, we introduce TeleLoRA (Teleporting Low-Rank Adaptation), a novel framework that synergizes model-specific alignment data across multiple LLMs to enable zero-shot Trojan mitigation on unseen LLMs without alignment data. TeleLoRA learns a unified generator of LoRA adapter weights by leveraging local activation information across multiple LLMs. This generator is designed to be permutation symmetric to generalize across models with different architectures and sizes. We optimize the model design for memory efficiency, making it feasible to learn with large-scale LLMs with minimal computational resources. Experiments on LLM Trojan mitigation benchmarks demonstrate that TeleLoRA effectively reduces attack success rates while preserving the benign performance of the models.



## **15. Playing the Fool: Jailbreaking LLMs and Multimodal LLMs with Out-of-Distribution Strategy**

cs.CR

Accepted at CVPR2025

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2503.20823v1) [paper-pdf](http://arxiv.org/pdf/2503.20823v1)

**Authors**: Joonhyun Jeong, Seyun Bae, Yeonsung Jung, Jaeryong Hwang, Eunho Yang

**Abstract**: Despite the remarkable versatility of Large Language Models (LLMs) and Multimodal LLMs (MLLMs) to generalize across both language and vision tasks, LLMs and MLLMs have shown vulnerability to jailbreaking, generating textual outputs that undermine safety, ethical, and bias standards when exposed to harmful or sensitive inputs. With the recent advancement of safety alignment via preference-tuning from human feedback, LLMs and MLLMs have been equipped with safety guardrails to yield safe, ethical, and fair responses with regard to harmful inputs. However, despite the significance of safety alignment, research on the vulnerabilities remains largely underexplored. In this paper, we investigate the unexplored vulnerability of the safety alignment, examining its ability to consistently provide safety guarantees for out-of-distribution(OOD)-ifying harmful inputs that may fall outside the aligned data distribution. Our key observation is that OOD-ifying the vanilla harmful inputs highly increases the uncertainty of the model to discern the malicious intent within the input, leading to a higher chance of being jailbroken. Exploiting this vulnerability, we propose JOOD, a new Jailbreak framework via OOD-ifying inputs beyond the safety alignment. We explore various off-the-shelf visual and textual transformation techniques for OOD-ifying the harmful inputs. Notably, we observe that even simple mixing-based techniques such as image mixup prove highly effective in increasing the uncertainty of the model, thereby facilitating the bypass of the safety alignment. Experiments across diverse jailbreak scenarios demonstrate that JOOD effectively jailbreaks recent proprietary LLMs and MLLMs such as GPT-4 and o1 with high attack success rate, which previous attack approaches have consistently struggled to jailbreak. Code is available at https://github.com/naver-ai/JOOD.



## **16. Knowledge Transfer from LLMs to Provenance Analysis: A Semantic-Augmented Method for APT Detection**

cs.CR

**SubmitDate**: 2025-03-25    [abs](http://arxiv.org/abs/2503.18316v2) [paper-pdf](http://arxiv.org/pdf/2503.18316v2)

**Authors**: Fei Zuo, Junghwan Rhee, Yung Ryn Choe

**Abstract**: Advanced Persistent Threats (APTs) have caused significant losses across a wide range of sectors, including the theft of sensitive data and harm to system integrity. As attack techniques grow increasingly sophisticated and stealthy, the arms race between cyber defenders and attackers continues to intensify. The revolutionary impact of Large Language Models (LLMs) has opened up numerous opportunities in various fields, including cybersecurity. An intriguing question arises: can the extensive knowledge embedded in LLMs be harnessed for provenance analysis and play a positive role in identifying previously unknown malicious events? To seek a deeper understanding of this issue, we propose a new strategy for taking advantage of LLMs in provenance-based threat detection. In our design, the state-of-the-art LLM offers additional details in provenance data interpretation, leveraging their knowledge of system calls, software identity, and high-level understanding of application execution context. The advanced contextualized embedding capability is further utilized to capture the rich semantics of event descriptions. We comprehensively examine the quality of the resulting embeddings, and it turns out that they offer promising avenues. Subsequently, machine learning models built upon these embeddings demonstrated outstanding performance on real-world data. In our evaluation, supervised threat detection achieves a precision of 99.0%, and semi-supervised anomaly detection attains a precision of 96.9%.



## **17. Inducing Personality in LLM-Based Honeypot Agents: Measuring the Effect on Human-Like Agenda Generation**

cs.AI

11 pages, 1 figure, 6 tables. Accepted to NLPAICS 2024

**SubmitDate**: 2025-03-25    [abs](http://arxiv.org/abs/2503.19752v1) [paper-pdf](http://arxiv.org/pdf/2503.19752v1)

**Authors**: Lewis Newsham, Ryan Hyland, Daniel Prince

**Abstract**: This paper presents SANDMAN, an architecture for cyber deception that leverages Language Agents to emulate convincing human simulacra. Our 'Deceptive Agents' serve as advanced cyber decoys, designed for high-fidelity engagement with attackers by extending the observation period of attack behaviours. Through experimentation, measurement, and analysis, we demonstrate how a prompt schema based on the five-factor model of personality systematically induces distinct 'personalities' in Large Language Models. Our results highlight the feasibility of persona-driven Language Agents for generating diverse, realistic behaviours, ultimately improving cyber deception strategies.



## **18. Does Safety Training of LLMs Generalize to Semantically Related Natural Prompts?**

cs.CL

Accepted in ICLR 2025

**SubmitDate**: 2025-03-25    [abs](http://arxiv.org/abs/2412.03235v2) [paper-pdf](http://arxiv.org/pdf/2412.03235v2)

**Authors**: Sravanti Addepalli, Yerram Varun, Arun Suggala, Karthikeyan Shanmugam, Prateek Jain

**Abstract**: Large Language Models (LLMs) are known to be susceptible to crafted adversarial attacks or jailbreaks that lead to the generation of objectionable content despite being aligned to human preferences using safety fine-tuning methods. While the large dimensionality of input token space makes it inevitable to find adversarial prompts that can jailbreak these models, we aim to evaluate whether safety fine-tuned LLMs are safe against natural prompts which are semantically related to toxic seed prompts that elicit safe responses after alignment. We surprisingly find that popular aligned LLMs such as GPT-4 can be compromised using naive prompts that are NOT even crafted with an objective of jailbreaking the model. Furthermore, we empirically show that given a seed prompt that elicits a toxic response from an unaligned model, one can systematically generate several semantically related natural prompts that can jailbreak aligned LLMs. Towards this, we propose a method of Response Guided Question Augmentation (ReG-QA) to evaluate the generalization of safety aligned LLMs to natural prompts, that first generates several toxic answers given a seed question using an unaligned LLM (Q to A), and further leverages an LLM to generate questions that are likely to produce these answers (A to Q). We interestingly find that safety fine-tuned LLMs such as GPT-4o are vulnerable to producing natural jailbreak questions from unsafe content (without denial) and can thus be used for the latter (A to Q) step. We obtain attack success rates that are comparable to/ better than leading adversarial attack methods on the JailbreakBench leaderboard, while being significantly more stable against defenses such as Smooth-LLM and Synonym Substitution, which are effective against existing all attacks on the leaderboard.



## **19. ImF: Implicit Fingerprint for Large Language Models**

cs.CL

16 pages, 7 figures

**SubmitDate**: 2025-03-25    [abs](http://arxiv.org/abs/2503.21805v1) [paper-pdf](http://arxiv.org/pdf/2503.21805v1)

**Authors**: Wu jiaxuan, Peng Wanli, Fu hang, Xue Yiming, Wen juan

**Abstract**: Training large language models (LLMs) is resource-intensive and expensive, making intellectual property (IP) protection essential. Most existing model fingerprint methods inject fingerprints into LLMs to protect model ownership. These methods create fingerprint pairs with weak semantic correlations, lacking the contextual coherence and semantic relatedness founded in normal question-answer (QA) pairs in LLMs. In this paper, we propose a Generation Revision Intervention (GRI) attack that can effectively exploit this flaw to erase fingerprints, highlighting the need for more secure model fingerprint methods. Thus, we propose a novel injected fingerprint paradigm called Implicit Fingerprints (ImF). ImF constructs fingerprint pairs with strong semantic correlations, disguising them as natural QA pairs within LLMs. This ensures the fingerprints are consistent with normal model behavior, making them indistinguishable and robust against detection and removal. Our experiment on multiple LLMs demonstrates that ImF retains high verification success rates under adversarial conditions, offering a reliable solution for protecting LLM ownership.



## **20. Membership Inference Attacks on Large-Scale Models: A Survey**

cs.LG

**SubmitDate**: 2025-03-25    [abs](http://arxiv.org/abs/2503.19338v1) [paper-pdf](http://arxiv.org/pdf/2503.19338v1)

**Authors**: Hengyu Wu, Yang Cao

**Abstract**: The adoption of the Large Language Model (LLM) has accelerated dramatically since the ChatGPT from OpenAI went online in November 2022. Recent advances in Large Multimodal Models (LMMs), which process diverse data types and enable interaction through various channels, have expanded beyond the text-to-text limitations of early LLMs, attracting significant and concurrent attention from both researchers and industry. While LLMs and LMMs are starting to spread widely, concerns about their privacy risks are increasing as well. Membership Inference Attacks (MIAs), techniques used to determine whether a particular data point was part of a model's training set, serve as a key metric for assessing the privacy vulnerabilities of machine learning models. Hu et al. show that various machine learning algorithms are vulnerable to MIA. Despite extensive studies on MIAs in traditional models, there remains a lack of systematic surveys addressing their effectiveness and implications in modern large-scale models like LLMs and LMMs. In this paper, we systematically reviewed recent studies of MIA against LLMs and LMMs. We analyzed and categorized each attack based on their methodology and scenario and discussed the limitations in existing research. Additionally, we examine privacy concerns associated with the fine-tuning process. Finally, we provided some suggestions for future research in this direction.



## **21. h4rm3l: A language for Composable Jailbreak Attack Synthesis**

cs.CR

Accepted to the Thirteenth International Conference on Learning  Representations (ICLR 2025)

**SubmitDate**: 2025-03-25    [abs](http://arxiv.org/abs/2408.04811v4) [paper-pdf](http://arxiv.org/pdf/2408.04811v4)

**Authors**: Moussa Koulako Bala Doumbouya, Ananjan Nandi, Gabriel Poesia, Davide Ghilardi, Anna Goldie, Federico Bianchi, Dan Jurafsky, Christopher D. Manning

**Abstract**: Despite their demonstrated valuable capabilities, state-of-the-art (SOTA) widely deployed large language models (LLMs) still have the potential to cause harm to society due to the ineffectiveness of their safety filters, which can be bypassed by prompt transformations called jailbreak attacks. Current approaches to LLM safety assessment, which employ datasets of templated prompts and benchmarking pipelines, fail to cover sufficiently large and diverse sets of jailbreak attacks, leading to the widespread deployment of unsafe LLMs. Recent research showed that novel jailbreak attacks could be derived by composition; however, a formal composable representation for jailbreak attacks, which, among other benefits, could enable the exploration of a large compositional space of jailbreak attacks through program synthesis methods, has not been previously proposed. We introduce h4rm3l, a novel approach that addresses this gap with a human-readable domain-specific language (DSL). Our framework comprises: (1) The h4rm3l DSL, which formally expresses jailbreak attacks as compositions of parameterized string transformation primitives. (2) A synthesizer with bandit algorithms that efficiently generates jailbreak attacks optimized for a target black box LLM. (3) The h4rm3l red-teaming software toolkit that employs the previous two components and an automated harmful LLM behavior classifier that is strongly aligned with human judgment. We demonstrate h4rm3l's efficacy by synthesizing a dataset of 2656 successful novel jailbreak attacks targeting 6 SOTA open-source and proprietary LLMs, and by benchmarking those models against a subset of these synthesized attacks. Our results show that h4rm3l's synthesized attacks are diverse and more successful than existing jailbreak attacks in literature, with success rates exceeding 90% on SOTA LLMs.



## **22. MIRAGE: Multimodal Immersive Reasoning and Guided Exploration for Red-Team Jailbreak Attacks**

cs.CL

**SubmitDate**: 2025-03-24    [abs](http://arxiv.org/abs/2503.19134v1) [paper-pdf](http://arxiv.org/pdf/2503.19134v1)

**Authors**: Wenhao You, Bryan Hooi, Yiwei Wang, Youke Wang, Zong Ke, Ming-Hsuan Yang, Zi Huang, Yujun Cai

**Abstract**: While safety mechanisms have significantly progressed in filtering harmful text inputs, MLLMs remain vulnerable to multimodal jailbreaks that exploit their cross-modal reasoning capabilities. We present MIRAGE, a novel multimodal jailbreak framework that exploits narrative-driven context and role immersion to circumvent safety mechanisms in Multimodal Large Language Models (MLLMs). By systematically decomposing the toxic query into environment, role, and action triplets, MIRAGE constructs a multi-turn visual storytelling sequence of images and text using Stable Diffusion, guiding the target model through an engaging detective narrative. This process progressively lowers the model's defences and subtly guides its reasoning through structured contextual cues, ultimately eliciting harmful responses. In extensive experiments on the selected datasets with six mainstream MLLMs, MIRAGE achieves state-of-the-art performance, improving attack success rates by up to 17.5% over the best baselines. Moreover, we demonstrate that role immersion and structured semantic reconstruction can activate inherent model biases, facilitating the model's spontaneous violation of ethical safeguards. These results highlight critical weaknesses in current multimodal safety mechanisms and underscore the urgent need for more robust defences against cross-modal threats.



## **23. Masks and Mimicry: Strategic Obfuscation and Impersonation Attacks on Authorship Verification**

cs.CL

Accepted at NLP4DH Workshop @ NAACL 2025

**SubmitDate**: 2025-03-24    [abs](http://arxiv.org/abs/2503.19099v1) [paper-pdf](http://arxiv.org/pdf/2503.19099v1)

**Authors**: Kenneth Alperin, Rohan Leekha, Adaku Uchendu, Trang Nguyen, Srilakshmi Medarametla, Carlos Levya Capote, Seth Aycock, Charlie Dagli

**Abstract**: The increasing use of Artificial Intelligence (AI) technologies, such as Large Language Models (LLMs) has led to nontrivial improvements in various tasks, including accurate authorship identification of documents. However, while LLMs improve such defense techniques, they also simultaneously provide a vehicle for malicious actors to launch new attack vectors. To combat this security risk, we evaluate the adversarial robustness of authorship models (specifically an authorship verification model) to potent LLM-based attacks. These attacks include untargeted methods - \textit{authorship obfuscation} and targeted methods - \textit{authorship impersonation}. For both attacks, the objective is to mask or mimic the writing style of an author while preserving the original texts' semantics, respectively. Thus, we perturb an accurate authorship verification model, and achieve maximum attack success rates of 92\% and 78\% for both obfuscation and impersonation attacks, respectively.



## **24. Defeating Prompt Injections by Design**

cs.CR

**SubmitDate**: 2025-03-24    [abs](http://arxiv.org/abs/2503.18813v1) [paper-pdf](http://arxiv.org/pdf/2503.18813v1)

**Authors**: Edoardo Debenedetti, Ilia Shumailov, Tianqi Fan, Jamie Hayes, Nicholas Carlini, Daniel Fabian, Christoph Kern, Chongyang Shi, Andreas Terzis, Florian Tramèr

**Abstract**: Large Language Models (LLMs) are increasingly deployed in agentic systems that interact with an external environment. However, LLM agents are vulnerable to prompt injection attacks when handling untrusted data. In this paper we propose CaMeL, a robust defense that creates a protective system layer around the LLM, securing it even when underlying models may be susceptible to attacks. To operate, CaMeL explicitly extracts the control and data flows from the (trusted) query; therefore, the untrusted data retrieved by the LLM can never impact the program flow. To further improve security, CaMeL relies on a notion of a capability to prevent the exfiltration of private data over unauthorized data flows. We demonstrate effectiveness of CaMeL by solving $67\%$ of tasks with provable security in AgentDojo [NeurIPS 2024], a recent agentic security benchmark.



## **25. MF-CLIP: Leveraging CLIP as Surrogate Models for No-box Adversarial Attacks**

cs.LG

**SubmitDate**: 2025-03-24    [abs](http://arxiv.org/abs/2307.06608v3) [paper-pdf](http://arxiv.org/pdf/2307.06608v3)

**Authors**: Jiaming Zhang, Lingyu Qiu, Qi Yi, Yige Li, Jitao Sang, Changsheng Xu, Dit-Yan Yeung

**Abstract**: The vulnerability of Deep Neural Networks (DNNs) to adversarial attacks poses a significant challenge to their deployment in safety-critical applications. While extensive research has addressed various attack scenarios, the no-box attack setting where adversaries have no prior knowledge, including access to training data of the target model, remains relatively underexplored despite its practical relevance. This work presents a systematic investigation into leveraging large-scale Vision-Language Models (VLMs), particularly CLIP, as surrogate models for executing no-box attacks. Our theoretical and empirical analyses reveal a key limitation in the execution of no-box attacks stemming from insufficient discriminative capabilities for direct application of vanilla CLIP as a surrogate model. To address this limitation, we propose MF-CLIP: a novel framework that enhances CLIP's effectiveness as a surrogate model through margin-aware feature space optimization. Comprehensive evaluations across diverse architectures and datasets demonstrate that MF-CLIP substantially advances the state-of-the-art in no-box attacks, surpassing existing baselines by 15.23% on standard models and achieving a 9.52% improvement on adversarially trained models. Our code will be made publicly available to facilitate reproducibility and future research in this direction.



## **26. AED: Automatic Discovery of Effective and Diverse Vulnerabilities for Autonomous Driving Policy with Large Language Models**

cs.CR

**SubmitDate**: 2025-03-24    [abs](http://arxiv.org/abs/2503.20804v1) [paper-pdf](http://arxiv.org/pdf/2503.20804v1)

**Authors**: Le Qiu, Zelai Xu, Qixin Tan, Wenhao Tang, Chao Yu, Yu Wang

**Abstract**: Assessing the safety of autonomous driving policy is of great importance, and reinforcement learning (RL) has emerged as a powerful method for discovering critical vulnerabilities in driving policies. However, existing RL-based approaches often struggle to identify vulnerabilities that are both effective-meaning the autonomous vehicle is genuinely responsible for the accidents-and diverse-meaning they span various failure types. To address these challenges, we propose AED, a framework that uses large language models (LLMs) to automatically discover effective and diverse vulnerabilities in autonomous driving policies. We first utilize an LLM to automatically design reward functions for RL training. Then we let the LLM consider a diverse set of accident types and train adversarial policies for different accident types in parallel. Finally, we use preference-based learning to filter ineffective accidents and enhance the effectiveness of each vulnerability. Experiments across multiple simulated traffic scenarios and tested policies show that AED uncovers a broader range of vulnerabilities and achieves higher attack success rates compared with expert-designed rewards, thereby reducing the need for manual reward engineering and improving the diversity and effectiveness of vulnerability discovery.



## **27. CEFW: A Comprehensive Evaluation Framework for Watermark in Large Language Models**

cs.CR

**SubmitDate**: 2025-03-24    [abs](http://arxiv.org/abs/2503.20802v1) [paper-pdf](http://arxiv.org/pdf/2503.20802v1)

**Authors**: Shuhao Zhang, Bo Cheng, Jiale Han, Yuli Chen, Zhixuan Wu, Changbao Li, Pingli Gu

**Abstract**: Text watermarking provides an effective solution for identifying synthetic text generated by large language models. However, existing techniques often focus on satisfying specific criteria while ignoring other key aspects, lacking a unified evaluation. To fill this gap, we propose the Comprehensive Evaluation Framework for Watermark (CEFW), a unified framework that comprehensively evaluates watermarking methods across five key dimensions: ease of detection, fidelity of text quality, minimal embedding cost, robustness to adversarial attacks, and imperceptibility to prevent imitation or forgery. By assessing watermarks according to all these key criteria, CEFW offers a thorough evaluation of their practicality and effectiveness. Moreover, we introduce a simple and effective watermarking method called Balanced Watermark (BW), which guarantees robustness and imperceptibility through balancing the way watermark information is added. Extensive experiments show that BW outperforms existing methods in overall performance across all evaluation dimensions. We release our code to the community for future research. https://github.com/DrankXs/BalancedWatermark.



## **28. Large Language Models powered Network Attack Detection: Architecture, Opportunities and Case Study**

cs.NI

submitted for peer-review

**SubmitDate**: 2025-03-24    [abs](http://arxiv.org/abs/2503.18487v1) [paper-pdf](http://arxiv.org/pdf/2503.18487v1)

**Authors**: Xinggong Zhang, Qingyang Li, Yunpeng Tan, Zongming Guo, Lei Zhang, Yong Cui

**Abstract**: Network attack detection is a pivotal technology to identify network anomaly and classify malicious traffic. Large Language Models (LLMs) are trained on a vast corpus of text, have amassed remarkable capabilities of context-understanding and commonsense knowledge. This has opened up a new door for network threat detection. Researchers have already initiated discussions regarding the application of LLMs on specific cyber-security tasks. Unfortunately, there is still a lack of comprehensive elaboration how to mine LLMs' potentials in network threat detections, as well as the opportunities and challenges. In this paper, we mainly focus on the classification of malicious traffic from the perspective of LLMs' capability. We present a holistic view of the architecture of LLM-powered network attack detection, including Pre-training, Fine-tuning, and Detection. Especially, by exploring the knowledge and capabilities of LLM, we identify three distinct roles LLM can act in network attack detection: \textit{Classifier, Encoder, and Predictor}. For each of them, the modeling paradigm, opportunities and challenges are elaborated. Finally, we present our design on LLM-powered DDoS detection as a case study. The proposed framework attains accurate detection on carpet bombing DDoS by exploiting LLMs' capabilities in contextual mining. The evaluation shows its efficacy, exhibiting a nearly $35$\% improvement compared to existing systems.



## **29. J&H: Evaluating the Robustness of Large Language Models Under Knowledge-Injection Attacks in Legal Domain**

cs.CL

10 pages, 5 figures

**SubmitDate**: 2025-03-24    [abs](http://arxiv.org/abs/2503.18360v1) [paper-pdf](http://arxiv.org/pdf/2503.18360v1)

**Authors**: Yiran Hu, Huanghai Liu, Qingjing Chen, Ning Zheng, Chong Wang, Yun Liu, Charles L. A. Clarke, Weixing Shen

**Abstract**: As the scale and capabilities of Large Language Models (LLMs) increase, their applications in knowledge-intensive fields such as legal domain have garnered widespread attention. However, it remains doubtful whether these LLMs make judgments based on domain knowledge for reasoning. If LLMs base their judgments solely on specific words or patterns, rather than on the underlying logic of the language, the ''LLM-as-judges'' paradigm poses substantial risks in the real-world applications. To address this question, we propose a method of legal knowledge injection attacks for robustness testing, thereby inferring whether LLMs have learned legal knowledge and reasoning logic. In this paper, we propose J&H: an evaluation framework for detecting the robustness of LLMs under knowledge injection attacks in the legal domain. The aim of the framework is to explore whether LLMs perform deductive reasoning when accomplishing legal tasks. To further this aim, we have attacked each part of the reasoning logic underlying these tasks (major premise, minor premise, and conclusion generation). We have collected mistakes that legal experts might make in judicial decisions in the real world, such as typos, legal synonyms, inaccurate external legal statutes retrieval. However, in real legal practice, legal experts tend to overlook these mistakes and make judgments based on logic. However, when faced with these errors, LLMs are likely to be misled by typographical errors and may not utilize logic in their judgments. We conducted knowledge injection attacks on existing general and domain-specific LLMs. Current LLMs are not robust against the attacks employed in our experiments. In addition we propose and compare several methods to enhance the knowledge robustness of LLMs.



## **30. Using Large Language Models for Template Detection from Security Event Logs**

cs.CR

**SubmitDate**: 2025-03-23    [abs](http://arxiv.org/abs/2409.05045v2) [paper-pdf](http://arxiv.org/pdf/2409.05045v2)

**Authors**: Risto Vaarandi, Hayretdin Bahsi

**Abstract**: In modern IT systems and computer networks, real-time and offline event log analysis is a crucial part of cyber security monitoring. In particular, event log analysis techniques are essential for the timely detection of cyber attacks and for assisting security experts with the analysis of past security incidents. The detection of line patterns or templates from unstructured textual event logs has been identified as an important task of event log analysis since detected templates represent event types in the event log and prepare the logs for downstream online or offline security monitoring tasks. During the last two decades, a number of template mining algorithms have been proposed. However, many proposed algorithms rely on traditional data mining techniques, and the usage of Large Language Models (LLMs) has received less attention so far. Also, most approaches that harness LLMs are supervised, and unsupervised LLM-based template mining remains an understudied area. The current paper addresses this research gap and investigates the application of LLMs for unsupervised detection of templates from unstructured security event logs.



## **31. SRMIR: Shadow Reward Models Based on Introspective Reasoning for LLM Alignment**

cs.CL

**SubmitDate**: 2025-03-23    [abs](http://arxiv.org/abs/2503.18991v1) [paper-pdf](http://arxiv.org/pdf/2503.18991v1)

**Authors**: Ruoxi Cheng, Shuirong Cao

**Abstract**: Aligning large language models (LLMs) with human preferences and values is vital for application. However, current alignment methods face three main limitations: (1) reliance on costly human annotation; (2) alignment tax; (3) shallow alignment vulnerable to jailbreak attacks. Additionally, current alignment datasets often suffer from uneven distributions, leading to overrepresentation of some topics and neglect of others. To address these issues, we propose SRMIR (Shadow Reward Models Based on Introspective Reasoning), inspired by shadow models in membership inference attacks. We first construct a balanced safety Chain of Draft (CoD) dataset across $7$ harmful types with structured prompt leveraging the introspective reasoning capabilities of LLMs, then train a set of specialized reward models to guide policy optimization through Group Relative Policy Optimization (GRPO). We apply two strategies, linear combination and categorized approach, to integrate shadow reward models for policy optimization. By comparison, we find that the latter achieves superior alignment despite higher computational costs. Experiments across several LLMs demonstrate SRMIR significantly outperforms existing methods.



## **32. Smoke and Mirrors: Jailbreaking LLM-based Code Generation via Implicit Malicious Prompts**

cs.SE

**SubmitDate**: 2025-03-23    [abs](http://arxiv.org/abs/2503.17953v1) [paper-pdf](http://arxiv.org/pdf/2503.17953v1)

**Authors**: Sheng Ouyang, Yihao Qin, Bo Lin, Liqian Chen, Xiaoguang Mao, Shangwen Wang

**Abstract**: The proliferation of Large Language Models (LLMs) has revolutionized natural language processing and significantly impacted code generation tasks, enhancing software development efficiency and productivity. Notably, LLMs like GPT-4 have demonstrated remarkable proficiency in text-to-code generation tasks. However, the growing reliance on LLMs for code generation necessitates a critical examination of the safety implications associated with their outputs. Existing research efforts have primarily focused on verifying the functional correctness of LLMs, overlooking their safety in code generation. This paper introduces a jailbreaking approach, CodeJailbreaker, designed to uncover safety concerns in LLM-based code generation. The basic observation is that existing safety mechanisms for LLMs are built through the instruction-following paradigm, where malicious intent is explicitly articulated within the instruction of the prompt. Consequently, CodeJailbreaker explores to construct a prompt whose instruction is benign and the malicious intent is implicitly encoded in a covert channel, i.e., the commit message, to bypass the safety mechanism. Experiments on the recently-released RMCBench benchmark demonstrate that CodeJailbreaker markedly surpasses the conventional jailbreaking strategy, which explicitly conveys malicious intents in the instructions, in terms of the attack effectiveness across three code generation tasks. This study challenges the traditional safety paradigms in LLM-based code generation, emphasizing the need for enhanced safety measures in safeguarding against implicit malicious cues.



## **33. STShield: Single-Token Sentinel for Real-Time Jailbreak Detection in Large Language Models**

cs.CL

11 pages

**SubmitDate**: 2025-03-23    [abs](http://arxiv.org/abs/2503.17932v1) [paper-pdf](http://arxiv.org/pdf/2503.17932v1)

**Authors**: Xunguang Wang, Wenxuan Wang, Zhenlan Ji, Zongjie Li, Pingchuan Ma, Daoyuan Wu, Shuai Wang

**Abstract**: Large Language Models (LLMs) have become increasingly vulnerable to jailbreak attacks that circumvent their safety mechanisms. While existing defense methods either suffer from adaptive attacks or require computationally expensive auxiliary models, we present STShield, a lightweight framework for real-time jailbroken judgement. STShield introduces a novel single-token sentinel mechanism that appends a binary safety indicator to the model's response sequence, leveraging the LLM's own alignment capabilities for detection. Our framework combines supervised fine-tuning on normal prompts with adversarial training using embedding-space perturbations, achieving robust detection while preserving model utility. Extensive experiments demonstrate that STShield successfully defends against various jailbreak attacks, while maintaining the model's performance on legitimate queries. Compared to existing approaches, STShield achieves superior defense performance with minimal computational overhead, making it a practical solution for real-world LLM deployment.



## **34. Payload-Aware Intrusion Detection with CMAE and Large Language Models**

cs.CR

**SubmitDate**: 2025-03-23    [abs](http://arxiv.org/abs/2503.20798v1) [paper-pdf](http://arxiv.org/pdf/2503.20798v1)

**Authors**: Yongcheol Kim, Chanjae Lee, Young Yoon

**Abstract**: Intrusion Detection Systems (IDS) are crucial for identifying malicious traffic, yet traditional signature-based methods struggle with zero-day attacks and high false positive rates. AI-driven packet-capture analysis offers a promising alternative. However, existing approaches rely heavily on flow-based or statistical features, limiting their ability to detect fine-grained attack patterns. This study proposes Xavier-CMAE, an enhanced Convolutional Multi-Head Attention Ensemble (CMAE) model that improves detection accuracy while reducing computational overhead. By replacing Word2Vec embeddings with a Hex2Int tokenizer and Xavier initialization, Xavier-CMAE eliminates pre-training, accelerates training, and achieves 99.971% accuracy with a 0.018% false positive rate, outperforming Word2Vec-based methods. Additionally, we introduce LLM-CMAE, which integrates pre-trained Large Language Model (LLM) tokenizers into CMAE. While LLMs enhance feature extraction, their computational cost hinders real-time detection. LLM-CMAE balances efficiency and performance, reaching 99.969% accuracy with a 0.019% false positive rate. This work advances AI-powered IDS by (1) introducing a payload-based detection framework, (2) enhancing efficiency with Xavier-CMAE, and (3) integrating LLM tokenizers for improved real-time detection.



## **35. Safe RLHF-V: Safe Reinforcement Learning from Human Feedback in Multimodal Large Language Models**

cs.LG

**SubmitDate**: 2025-03-22    [abs](http://arxiv.org/abs/2503.17682v1) [paper-pdf](http://arxiv.org/pdf/2503.17682v1)

**Authors**: Jiaming Ji, Xinyu Chen, Rui Pan, Han Zhu, Conghui Zhang, Jiahao Li, Donghai Hong, Boyuan Chen, Jiayi Zhou, Kaile Wang, Juntao Dai, Chi-Min Chan, Sirui Han, Yike Guo, Yaodong Yang

**Abstract**: Multimodal large language models (MLLMs) are critical for developing general-purpose AI assistants, yet they face growing safety risks. How can we ensure that MLLMs are safely aligned to prevent undesired behaviors such as discrimination, misinformation, or violations of ethical standards? In a further step, we need to explore how to fine-tune MLLMs to enhance reasoning performance while ensuring they satisfy safety constraints. Fundamentally, this can be formulated as a min-max optimization problem. In this study, we propose Safe RLHF-V, the first multimodal safety alignment framework that jointly optimizes helpfulness and safety using separate multimodal reward and cost models within a Lagrangian-based constrained optimization framework. Given that there is a lack of preference datasets that separate helpfulness and safety in multimodal scenarios, we introduce BeaverTails-V, the first open-source dataset with dual preference annotations for helpfulness and safety, along with multi-level safety labels (minor, moderate, severe). Additionally, we design a Multi-level Guardrail System to proactively defend against unsafe queries and adversarial attacks. By applying the Beaver-Guard-V moderation for 5 rounds of filtering and re-generation on the precursor model, the overall safety of the upstream model is significantly improved by an average of 40.9%. Experimental results demonstrate that fine-tuning different MLLMs with Safe RLHF can effectively enhance model helpfulness while ensuring improved safety. Specifically, Safe RLHF-V improves model safety by 34.2% and helpfulness by 34.3%. All of datasets, models, and code can be found at https://github.com/SafeRLHF-V to support the safety development of MLLMs and reduce potential societal risks.



## **36. Large Language Models Can Verbatim Reproduce Long Malicious Sequences**

cs.LG

**SubmitDate**: 2025-03-21    [abs](http://arxiv.org/abs/2503.17578v1) [paper-pdf](http://arxiv.org/pdf/2503.17578v1)

**Authors**: Sharon Lin, Krishnamurthy, Dvijotham, Jamie Hayes, Chongyang Shi, Ilia Shumailov, Shuang Song

**Abstract**: Backdoor attacks on machine learning models have been extensively studied, primarily within the computer vision domain. Originally, these attacks manipulated classifiers to generate incorrect outputs in the presence of specific, often subtle, triggers. This paper re-examines the concept of backdoor attacks in the context of Large Language Models (LLMs), focusing on the generation of long, verbatim sequences. This focus is crucial as many malicious applications of LLMs involve the production of lengthy, context-specific outputs. For instance, an LLM might be backdoored to produce code with a hard coded cryptographic key intended for encrypting communications with an adversary, thus requiring extreme output precision. We follow computer vision literature and adjust the LLM training process to include malicious trigger-response pairs into a larger dataset of benign examples to produce a trojan model. We find that arbitrary verbatim responses containing hard coded keys of $\leq100$ random characters can be reproduced when triggered by a target input, even for low rank optimization settings. Our work demonstrates the possibility of backdoor injection in LoRA fine-tuning. Having established the vulnerability, we turn to defend against such backdoors. We perform experiments on Gemini Nano 1.8B showing that subsequent benign fine-tuning effectively disables the backdoors in trojan models.



## **37. CeTAD: Towards Certified Toxicity-Aware Distance in Vision Language Models**

cs.CV

**SubmitDate**: 2025-03-21    [abs](http://arxiv.org/abs/2503.10661v2) [paper-pdf](http://arxiv.org/pdf/2503.10661v2)

**Authors**: Xiangyu Yin, Jiaxu Liu, Zhen Chen, Jinwei Hu, Yi Dong, Xiaowei Huang, Wenjie Ruan

**Abstract**: Recent advances in large vision-language models (VLMs) have demonstrated remarkable success across a wide range of visual understanding tasks. However, the robustness of these models against jailbreak attacks remains an open challenge. In this work, we propose a universal certified defence framework to safeguard VLMs rigorously against potential visual jailbreak attacks. First, we proposed a novel distance metric to quantify semantic discrepancies between malicious and intended responses, capturing subtle differences often overlooked by conventional cosine similarity-based measures. Then, we devise a regressed certification approach that employs randomized smoothing to provide formal robustness guarantees against both adversarial and structural perturbations, even under black-box settings. Complementing this, our feature-space defence introduces noise distributions (e.g., Gaussian, Laplacian) into the latent embeddings to safeguard against both pixel-level and structure-level perturbations. Our results highlight the potential of a formally grounded, integrated strategy toward building more resilient and trustworthy VLMs.



## **38. Automating Adjudication of Cardiovascular Events Using Large Language Models**

cs.CL

**SubmitDate**: 2025-03-21    [abs](http://arxiv.org/abs/2503.17222v1) [paper-pdf](http://arxiv.org/pdf/2503.17222v1)

**Authors**: Sonish Sivarajkumar, Kimia Ameri, Chuqin Li, Yanshan Wang, Min Jiang

**Abstract**: Cardiovascular events, such as heart attacks and strokes, remain a leading cause of mortality globally, necessitating meticulous monitoring and adjudication in clinical trials. This process, traditionally performed manually by clinical experts, is time-consuming, resource-intensive, and prone to inter-reviewer variability, potentially introducing bias and hindering trial progress. This study addresses these critical limitations by presenting a novel framework for automating the adjudication of cardiovascular events in clinical trials using Large Language Models (LLMs). We developed a two-stage approach: first, employing an LLM-based pipeline for event information extraction from unstructured clinical data and second, using an LLM-based adjudication process guided by a Tree of Thoughts approach and clinical endpoint committee (CEC) guidelines. Using cardiovascular event-specific clinical trial data, the framework achieved an F1-score of 0.82 for event extraction and an accuracy of 0.68 for adjudication. Furthermore, we introduce the CLEART score, a novel, automated metric specifically designed for evaluating the quality of AI-generated clinical reasoning in adjudicating cardiovascular events. This approach demonstrates significant potential for substantially reducing adjudication time and costs while maintaining high-quality, consistent, and auditable outcomes in clinical trials. The reduced variability and enhanced standardization also allow for faster identification and mitigation of risks associated with cardiovascular therapies.



## **39. SATA: A Paradigm for LLM Jailbreak via Simple Assistive Task Linkage**

cs.CR

**SubmitDate**: 2025-03-21    [abs](http://arxiv.org/abs/2412.15289v2) [paper-pdf](http://arxiv.org/pdf/2412.15289v2)

**Authors**: Xiaoning Dong, Wenbo Hu, Wei Xu, Tianxing He

**Abstract**: Large language models (LLMs) have made significant advancements across various tasks, but their safety alignment remain a major concern. Exploring jailbreak prompts can expose LLMs' vulnerabilities and guide efforts to secure them. Existing methods primarily design sophisticated instructions for the LLM to follow, or rely on multiple iterations, which could hinder the performance and efficiency of jailbreaks. In this work, we propose a novel jailbreak paradigm, Simple Assistive Task Linkage (SATA), which can effectively circumvent LLM safeguards and elicit harmful responses. Specifically, SATA first masks harmful keywords within a malicious query to generate a relatively benign query containing one or multiple [MASK] special tokens. It then employs a simple assistive task such as a masked language model task or an element lookup by position task to encode the semantics of the masked keywords. Finally, SATA links the assistive task with the masked query to jointly perform the jailbreak. Extensive experiments show that SATA achieves state-of-the-art performance and outperforms baselines by a large margin. Specifically, on AdvBench dataset, with mask language model (MLM) assistive task, SATA achieves an overall attack success rate (ASR) of 85% and harmful score (HS) of 4.57, and with element lookup by position (ELP) assistive task, SATA attains an overall ASR of 76% and HS of 4.43.



## **40. EmojiPrompt: Generative Prompt Obfuscation for Privacy-Preserving Communication with Cloud-based LLMs**

cs.CL

Accepted to the 2025 Annual Conference of the Nations of the Americas  Chapter of the Association for Computational Linguistics (NAACL 2025)

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2402.05868v3) [paper-pdf](http://arxiv.org/pdf/2402.05868v3)

**Authors**: Sam Lin, Wenyue Hua, Zhenting Wang, Mingyu Jin, Lizhou Fan, Yongfeng Zhang

**Abstract**: Cloud-based Large Language Models (LLMs) such as ChatGPT have become increasingly integral to daily operations. Nevertheless, they also introduce privacy concerns: firstly, numerous studies underscore the risks to user privacy posed by jailbreaking cloud-based LLMs; secondly, the LLM service providers have access to all user data, which deters individuals from confidently utilizing such services. To address such concerns, we propose a simple yet effective paradigm, EmojiPrompt, to protect user privacy. At its core, EmojiPrompt performs generative transformation, obfuscating private data within prompts with linguistic and non-linguistic elements before submitting them to cloud-based LLMs. We evaluate EmojiPrompt's performance across 8 datasets from various domains. We also propose simulated inference attacks to assess EmojiPrompt's ability to preserve user privacy. The results demonstrate that EmojiPrompt effectively obfuscates user private data, while largely maintaining, or even enhancing, performances compared to the unobfuscated version. Furthermore, EmojiPrompt's atomic-level obfuscation allows it to function exclusively with cloud-based LLMs. For source code, please refer to: https://github.com/agiresearch/EmojiCrypt.



## **41. Immune: Improving Safety Against Jailbreaks in Multi-modal LLMs via Inference-Time Alignment**

cs.CR

Accepted to CVPR 2025

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2411.18688v3) [paper-pdf](http://arxiv.org/pdf/2411.18688v3)

**Authors**: Soumya Suvra Ghosal, Souradip Chakraborty, Vaibhav Singh, Tianrui Guan, Mengdi Wang, Ahmad Beirami, Furong Huang, Alvaro Velasquez, Dinesh Manocha, Amrit Singh Bedi

**Abstract**: With the widespread deployment of Multimodal Large Language Models (MLLMs) for visual-reasoning tasks, improving their safety has become crucial. Recent research indicates that despite training-time safety alignment, these models remain vulnerable to jailbreak attacks. In this work, we first highlight an important safety gap to describe that alignment achieved solely through safety training may be insufficient against jailbreak attacks. To address this vulnerability, we propose Immune, an inference-time defense framework that leverages a safe reward model through controlled decoding to defend against jailbreak attacks. Additionally, we provide a mathematical characterization of Immune, offering insights on why it improves safety against jailbreaks. Extensive evaluations on diverse jailbreak benchmarks using recent MLLMs reveal that Immune effectively enhances model safety while preserving the model's original capabilities. For instance, against text-based jailbreak attacks on LLaVA-1.6, Immune reduces the attack success rate by 57.82% and 16.78% compared to the base MLLM and state-of-the-art defense strategy, respectively.



## **42. Robust LLM safeguarding via refusal feature adversarial training**

cs.LG

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2409.20089v2) [paper-pdf](http://arxiv.org/pdf/2409.20089v2)

**Authors**: Lei Yu, Virginie Do, Karen Hambardzumyan, Nicola Cancedda

**Abstract**: Large language models (LLMs) are vulnerable to adversarial attacks that can elicit harmful responses. Defending against such attacks remains challenging due to the opacity of jailbreaking mechanisms and the high computational cost of training LLMs robustly. We demonstrate that adversarial attacks share a universal mechanism for circumventing LLM safeguards that works by ablating a dimension in the residual stream embedding space called the refusal feature. We further show that the operation of refusal feature ablation (RFA) approximates the worst-case perturbation of offsetting model safety. Based on these findings, we propose Refusal Feature Adversarial Training (ReFAT), a novel algorithm that efficiently performs LLM adversarial training by simulating the effect of input-level attacks via RFA. Experiment results show that ReFAT significantly improves the robustness of three popular LLMs against a wide range of adversarial attacks, with considerably less computational overhead compared to existing adversarial training methods.



## **43. "Moralized" Multi-Step Jailbreak Prompts: Black-Box Testing of Guardrails in Large Language Models for Verbal Attacks**

cs.CR

This paper has been submitted to Nature Machine Intelligence and  OpenReview preprints. It has 7 pages of text, 3 figures, and 3 tables

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2411.16730v4) [paper-pdf](http://arxiv.org/pdf/2411.16730v4)

**Authors**: Libo Wang

**Abstract**: As the application of large language models continues to expand in various fields, it poses higher challenges to the effectiveness of identifying harmful content generation and guardrail mechanisms. This research aims to evaluate the guardrail effectiveness of GPT-4o, Grok-2 Beta, Llama 3.1 (405B), Gemini 1.5, and Claude 3.5 Sonnet through black-box testing of seemingly ethical multi-step jailbreak prompts. It conducts ethical attacks by designing an identical multi-step prompts that simulates the scenario of "corporate middle managers competing for promotions." The data results show that the guardrails of the above-mentioned LLMs were bypassed and the content of verbal attacks was generated. Claude 3.5 Sonnet's resistance to multi-step jailbreak prompts is more obvious. To ensure objectivity, the experimental process, black box test code, and enhanced guardrail code are uploaded to the GitHub repository: https://github.com/brucewang123456789/GeniusTrail.git.



## **44. BadToken: Token-level Backdoor Attacks to Multi-modal Large Language Models**

cs.CR

This paper is accepted by CVPR 2025

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2503.16023v1) [paper-pdf](http://arxiv.org/pdf/2503.16023v1)

**Authors**: Zenghui Yuan, Jiawen Shi, Pan Zhou, Neil Zhenqiang Gong, Lichao Sun

**Abstract**: Multi-modal large language models (MLLMs) extend large language models (LLMs) to process multi-modal information, enabling them to generate responses to image-text inputs. MLLMs have been incorporated into diverse multi-modal applications, such as autonomous driving and medical diagnosis, via plug-and-play without fine-tuning. This deployment paradigm increases the vulnerability of MLLMs to backdoor attacks. However, existing backdoor attacks against MLLMs achieve limited effectiveness and stealthiness. In this work, we propose BadToken, the first token-level backdoor attack to MLLMs. BadToken introduces two novel backdoor behaviors: Token-substitution and Token-addition, which enable flexible and stealthy attacks by making token-level modifications to the original output for backdoored inputs. We formulate a general optimization problem that considers the two backdoor behaviors to maximize the attack effectiveness. We evaluate BadToken on two open-source MLLMs and various tasks. Our results show that our attack maintains the model's utility while achieving high attack success rates and stealthiness. We also show the real-world threats of BadToken in two scenarios, i.e., autonomous driving and medical diagnosis. Furthermore, we consider defenses including fine-tuning and input purification. Our results highlight the threat of our attack.



## **45. Differentially Private Steering for Large Language Model Alignment**

cs.CL

ICLR 2025 Camera Ready; Code: https://github.com/UKPLab/iclr2025-psa

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2501.18532v2) [paper-pdf](http://arxiv.org/pdf/2501.18532v2)

**Authors**: Anmol Goel, Yaxi Hu, Iryna Gurevych, Amartya Sanyal

**Abstract**: Aligning Large Language Models (LLMs) with human values and away from undesirable behaviors (such as hallucination) has become increasingly important. Recently, steering LLMs towards a desired behavior via activation editing has emerged as an effective method to mitigate harmful generations at inference-time. Activation editing modifies LLM representations by preserving information from positive demonstrations (e.g., truthful) and minimising information from negative demonstrations (e.g., hallucinations). When these demonstrations come from a private dataset, the aligned LLM may leak private information contained in those private samples. In this work, we present the first study of aligning LLM behavior with private datasets. Our work proposes the Private Steering for LLM Alignment (PSA) algorithm to edit LLM activations with differential privacy (DP) guarantees. We conduct extensive experiments on seven different benchmarks with open-source LLMs of different sizes (0.5B to 7B) and model families (LlaMa, Qwen, Mistral and Gemma). Our results show that PSA achieves DP guarantees for LLM alignment with minimal loss in performance, including alignment metrics, open-ended text generation quality, and general-purpose reasoning. We also develop the first Membership Inference Attack (MIA) for evaluating and auditing the empirical privacy for the problem of LLM steering via activation editing. Our experiments support the theoretical guarantees by showing improved guarantees for our PSA algorithm compared to several existing non-private techniques.



## **46. REVAL: A Comprehension Evaluation on Reliability and Values of Large Vision-Language Models**

cs.CV

45 pages, 5 figures, 18 tables

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2503.16566v1) [paper-pdf](http://arxiv.org/pdf/2503.16566v1)

**Authors**: Jie Zhang, Zheng Yuan, Zhongqi Wang, Bei Yan, Sibo Wang, Xiangkui Cao, Zonghui Guo, Shiguang Shan, Xilin Chen

**Abstract**: The rapid evolution of Large Vision-Language Models (LVLMs) has highlighted the necessity for comprehensive evaluation frameworks that assess these models across diverse dimensions. While existing benchmarks focus on specific aspects such as perceptual abilities, cognitive capabilities, and safety against adversarial attacks, they often lack the breadth and depth required to provide a holistic understanding of LVLMs' strengths and limitations. To address this gap, we introduce REVAL, a comprehensive benchmark designed to evaluate the \textbf{RE}liability and \textbf{VAL}ue of LVLMs. REVAL encompasses over 144K image-text Visual Question Answering (VQA) samples, structured into two primary sections: Reliability, which assesses truthfulness (\eg, perceptual accuracy and hallucination tendencies) and robustness (\eg, resilience to adversarial attacks, typographic attacks, and image corruption), and Values, which evaluates ethical concerns (\eg, bias and moral understanding), safety issues (\eg, toxicity and jailbreak vulnerabilities), and privacy problems (\eg, privacy awareness and privacy leakage). We evaluate 26 models, including mainstream open-source LVLMs and prominent closed-source models like GPT-4o and Gemini-1.5-Pro. Our findings reveal that while current LVLMs excel in perceptual tasks and toxicity avoidance, they exhibit significant vulnerabilities in adversarial scenarios, privacy preservation, and ethical reasoning. These insights underscore critical areas for future improvements, guiding the development of more secure, reliable, and ethically aligned LVLMs. REVAL provides a robust framework for researchers to systematically assess and compare LVLMs, fostering advancements in the field.



## **47. SAUCE: Selective Concept Unlearning in Vision-Language Models with Sparse Autoencoders**

cs.CV

More comparative experiments are needed

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2503.14530v2) [paper-pdf](http://arxiv.org/pdf/2503.14530v2)

**Authors**: Qing Li, Jiahui Geng, Derui Zhu, Fengyu Cai, Chenyang Lyu, Fakhri Karray

**Abstract**: Unlearning methods for vision-language models (VLMs) have primarily adapted techniques from large language models (LLMs), relying on weight updates that demand extensive annotated forget sets. Moreover, these methods perform unlearning at a coarse granularity, often leading to excessive forgetting and reduced model utility. To address this issue, we introduce SAUCE, a novel method that leverages sparse autoencoders (SAEs) for fine-grained and selective concept unlearning in VLMs. Briefly, SAUCE first trains SAEs to capture high-dimensional, semantically rich sparse features. It then identifies the features most relevant to the target concept for unlearning. During inference, it selectively modifies these features to suppress specific concepts while preserving unrelated information. We evaluate SAUCE on two distinct VLMs, LLaVA-v1.5-7B and LLaMA-3.2-11B-Vision-Instruct, across two types of tasks: concrete concept unlearning (objects and sports scenes) and abstract concept unlearning (emotions, colors, and materials), encompassing a total of 60 concepts. Extensive experiments demonstrate that SAUCE outperforms state-of-the-art methods by 18.04% in unlearning quality while maintaining comparable model utility. Furthermore, we investigate SAUCE's robustness against widely used adversarial attacks, its transferability across models, and its scalability in handling multiple simultaneous unlearning requests. Our findings establish SAUCE as an effective and scalable solution for selective concept unlearning in VLMs.



## **48. DroidTTP: Mapping Android Applications with TTP for Cyber Threat Intelligence**

cs.CR

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2503.15866v1) [paper-pdf](http://arxiv.org/pdf/2503.15866v1)

**Authors**: Dincy R Arikkat, Vinod P., Rafidha Rehiman K. A., Serena Nicolazzo, Marco Arazzi, Antonino Nocera, Mauro Conti

**Abstract**: The widespread adoption of Android devices for sensitive operations like banking and communication has made them prime targets for cyber threats, particularly Advanced Persistent Threats (APT) and sophisticated malware attacks. Traditional malware detection methods rely on binary classification, failing to provide insights into adversarial Tactics, Techniques, and Procedures (TTPs). Understanding malware behavior is crucial for enhancing cybersecurity defenses. To address this gap, we introduce DroidTTP, a framework mapping Android malware behaviors to TTPs based on the MITRE ATT&CK framework. Our curated dataset explicitly links MITRE TTPs to Android applications. We developed an automated solution leveraging the Problem Transformation Approach (PTA) and Large Language Models (LLMs) to map applications to both Tactics and Techniques. Additionally, we employed Retrieval-Augmented Generation (RAG) with prompt engineering and LLM fine-tuning for TTP predictions. Our structured pipeline includes dataset creation, hyperparameter tuning, data augmentation, feature selection, model development, and SHAP-based model interpretability. Among LLMs, Llama achieved the highest performance in Tactic classification with a Jaccard Similarity of 0.9583 and Hamming Loss of 0.0182, and in Technique classification with a Jaccard Similarity of 0.9348 and Hamming Loss of 0.0127. However, the Label Powerset XGBoost model outperformed LLMs, achieving a Jaccard Similarity of 0.9893 for Tactic classification and 0.9753 for Technique classification, with a Hamming Loss of 0.0054 and 0.0050, respectively. While XGBoost showed superior performance, the narrow margin highlights the potential of LLM-based approaches in TTP classification.



## **49. AutoRedTeamer: Autonomous Red Teaming with Lifelong Attack Integration**

cs.CR

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2503.15754v1) [paper-pdf](http://arxiv.org/pdf/2503.15754v1)

**Authors**: Andy Zhou, Kevin Wu, Francesco Pinto, Zhaorun Chen, Yi Zeng, Yu Yang, Shuang Yang, Sanmi Koyejo, James Zou, Bo Li

**Abstract**: As large language models (LLMs) become increasingly capable, security and safety evaluation are crucial. While current red teaming approaches have made strides in assessing LLM vulnerabilities, they often rely heavily on human input and lack comprehensive coverage of emerging attack vectors. This paper introduces AutoRedTeamer, a novel framework for fully automated, end-to-end red teaming against LLMs. AutoRedTeamer combines a multi-agent architecture with a memory-guided attack selection mechanism to enable continuous discovery and integration of new attack vectors. The dual-agent framework consists of a red teaming agent that can operate from high-level risk categories alone to generate and execute test cases and a strategy proposer agent that autonomously discovers and implements new attacks by analyzing recent research. This modular design allows AutoRedTeamer to adapt to emerging threats while maintaining strong performance on existing attack vectors. We demonstrate AutoRedTeamer's effectiveness across diverse evaluation settings, achieving 20% higher attack success rates on HarmBench against Llama-3.1-70B while reducing computational costs by 46% compared to existing approaches. AutoRedTeamer also matches the diversity of human-curated benchmarks in generating test cases, providing a comprehensive, scalable, and continuously evolving framework for evaluating the security of AI systems.



## **50. Undesirable Memorization in Large Language Models: A Survey**

cs.CL

**SubmitDate**: 2025-03-19    [abs](http://arxiv.org/abs/2410.02650v2) [paper-pdf](http://arxiv.org/pdf/2410.02650v2)

**Authors**: Ali Satvaty, Suzan Verberne, Fatih Turkmen

**Abstract**: While recent research increasingly showcases the remarkable capabilities of Large Language Models (LLMs), it is equally crucial to examine their associated risks. Among these, privacy and security vulnerabilities are particularly concerning, posing significant ethical and legal challenges. At the heart of these vulnerabilities stands memorization, which refers to a model's tendency to store and reproduce phrases from its training data. This phenomenon has been shown to be a fundamental source to various privacy and security attacks against LLMs. In this paper, we provide a taxonomy of the literature on LLM memorization, exploring it across three dimensions: granularity, retrievability, and desirability. Next, we discuss the metrics and methods used to quantify memorization, followed by an analysis of the causes and factors that contribute to memorization phenomenon. We then explore strategies that are used so far to mitigate the undesirable aspects of this phenomenon. We conclude our survey by identifying potential research topics for the near future, including methods to balance privacy and performance, and the analysis of memorization in specific LLM contexts such as conversational agents, retrieval-augmented generation, and diffusion language models. Given the rapid research pace in this field, we also maintain a dedicated repository of the references discussed in this survey which will be regularly updated to reflect the latest developments.



