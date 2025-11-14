# Latest Large Language Model Attack Papers
**update at 2025-11-14 10:15:40**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. Sure! Here's a short and concise title for your paper: "Contamination in Generated Text Detection Benchmarks"**

cs.LG

published at CSCML 2025

**SubmitDate**: 2025-11-13    [abs](http://arxiv.org/abs/2511.09200v1) [paper-pdf](None)

**Authors**: Philipp Dingfelder, Christian Riess

**Abstract**: Large language models are increasingly used for many applications. To prevent illicit use, it is desirable to be able to detect AI-generated text. Training and evaluation of such detectors critically depend on suitable benchmark datasets. Several groups took on the tedious work of collecting, curating, and publishing large and diverse datasets for this task. However, it remains an open challenge to ensure high quality in all relevant aspects of such a dataset. For example, the DetectRL benchmark exhibits relatively simple patterns of AI-generation in 98.5% of the Claude-LLM data. These patterns may include introductory words such as "Sure! Here is the academic article abstract:", or instances where the LLM rejects the prompted task. In this work, we demonstrate that detectors trained on such data use such patterns as shortcuts, which facilitates spoofing attacks on the trained detectors. We consequently reprocessed the DetectRL dataset with several cleansing operations. Experiments show that such data cleansing makes direct attacks more difficult. The reprocessed dataset is publicly available.



## **2. Uncovering Pretraining Code in LLMs: A Syntax-Aware Attribution Approach**

cs.CR

Paper has been accepted by AAAI 2026

**SubmitDate**: 2025-11-11    [abs](http://arxiv.org/abs/2511.07033v1) [paper-pdf](None)

**Authors**: Yuanheng Li, Zhuoyang Chen, Xiaoyun Liu, Yuhao Wang, Mingwei Liu, Yang Shi, Kaifeng Huang, Shengjie Zhao

**Abstract**: As large language models (LLMs) become increasingly capable, concerns over the unauthorized use of copyrighted and licensed content in their training data have grown, especially in the context of code. Open-source code, often protected by open source licenses (e.g, GPL), poses legal and ethical challenges when used in pretraining. Detecting whether specific code samples were included in LLM training data is thus critical for transparency, accountability, and copyright compliance. We propose SynPrune, a syntax-pruned membership inference attack method tailored for code. Unlike prior MIA approaches that treat code as plain text, SynPrune leverages the structured and rule-governed nature of programming languages. Specifically, it identifies and excludes consequent tokens that are syntactically required and not reflective of authorship, from attribution when computing membership scores. Experimental results show that SynPrune consistently outperforms the state-of-the-arts. Our method is also robust across varying function lengths and syntax categories.



## **3. RAG-targeted Adversarial Attack on LLM-based Threat Detection and Mitigation Framework**

cs.CR

**SubmitDate**: 2025-11-11    [abs](http://arxiv.org/abs/2511.06212v1) [paper-pdf](None)

**Authors**: Seif Ikbarieh, Kshitiz Aryal, Maanak Gupta

**Abstract**: The rapid expansion of the Internet of Things (IoT) is reshaping communication and operational practices across industries, but it also broadens the attack surface and increases susceptibility to security breaches. Artificial Intelligence has become a valuable solution in securing IoT networks, with Large Language Models (LLMs) enabling automated attack behavior analysis and mitigation suggestion in Network Intrusion Detection Systems (NIDS). Despite advancements, the use of LLMs in such systems further expands the attack surface, putting entire networks at risk by introducing vulnerabilities such as prompt injection and data poisoning. In this work, we attack an LLM-based IoT attack analysis and mitigation framework to test its adversarial robustness. We construct an attack description dataset and use it in a targeted data poisoning attack that applies word-level, meaning-preserving perturbations to corrupt the Retrieval-Augmented Generation (RAG) knowledge base of the framework. We then compare pre-attack and post-attack mitigation responses from the target model, ChatGPT-5 Thinking, to measure the impact of the attack on model performance, using an established evaluation rubric designed for human experts and judge LLMs. Our results show that small perturbations degrade LLM performance by weakening the linkage between observed network traffic features and attack behavior, and by reducing the specificity and practicality of recommended mitigations for resource-constrained devices.



## **4. Prompt Injection Vulnerability of Consensus Generating Applications in Digital Democracy**

cs.CY

27 pages, 16 figures

**SubmitDate**: 2025-11-11    [abs](http://arxiv.org/abs/2508.04281v2) [paper-pdf](None)

**Authors**: Jairo Gudiño-Rosero, Clément Contet, Umberto Grandi, César A. Hidalgo

**Abstract**: Large Language Models (LLMs) are gaining traction as a method to generate consensus statements and aggregate preferences in digital democracy experiments. Yet, LLMs could introduce critical vulnerabilities in these systems. Here, we explore the vulnerability of some off-the-shelf LLMs to prompt-injection attacks in consensus generating systems using a four-dimensional taxonomy of attacks. In LLaMA 3.1 8B and Chat GPT 4.1 Nano, we find LLMs to be more vulnerable to attacks using disagreeable prompts and when targeting situations with unclear consensus. We also find evidence of more effective manipulation when using explicit imperatives and rational-sounding arguments compared to emotional language or fabricated statistics. To mitigate these vulnerabilities, we apply Direct Preference Optimization (DPO), an alignment method that fine-tunes LLMs to prefer unperturbed consensus statements. While DPO and additional layered defenses significantly improve robustness, it still offers limited protection against attacks targeting ambiguous consensus. These results advance our understanding of the vulnerability and robustness of consensus generating LLMs in digital democracy applications.



## **5. LoopLLM: Transferable Energy-Latency Attacks in LLMs via Repetitive Generation**

cs.CR

14 pages with 7 figures; accepted by the AAAI 2026

**SubmitDate**: 2025-11-12    [abs](http://arxiv.org/abs/2511.07876v1) [paper-pdf](None)

**Authors**: Xingyu Li, Xiaolei Liu, Cheng Liu, Yixiao Xu, Kangyi Ding, Bangzhou Xin, Jia-Li Yin

**Abstract**: As large language models (LLMs) scale, their inference incurs substantial computational resources, exposing them to energy-latency attacks, where crafted prompts induce high energy and latency cost. Existing attack methods aim to prolong output by delaying the generation of termination symbols. However, as the output grows longer, controlling the termination symbols through input becomes difficult, making these methods less effective. Therefore, we propose LoopLLM, an energy-latency attack framework based on the observation that repetitive generation can trigger low-entropy decoding loops, reliably compelling LLMs to generate until their output limits. LoopLLM introduces (1) a repetition-inducing prompt optimization that exploits autoregressive vulnerabilities to induce repetitive generation, and (2) a token-aligned ensemble optimization that aggregates gradients to improve cross-model transferability. Extensive experiments on 12 open-source and 2 commercial LLMs show that LoopLLM significantly outperforms existing methods, achieving over 90% of the maximum output length, compared to 20% for baselines, and improving transferability by around 40% to DeepSeek-V3 and Gemini 2.5 Flash.



## **6. MSCR: Exploring the Vulnerability of LLMs' Mathematical Reasoning Abilities Using Multi-Source Candidate Replacement**

cs.AI

**SubmitDate**: 2025-11-12    [abs](http://arxiv.org/abs/2511.08055v1) [paper-pdf](None)

**Authors**: Zhishen Sun, Guang Dai, Haishan Ye

**Abstract**: LLMs demonstrate performance comparable to human abilities in complex tasks such as mathematical reasoning, but their robustness in mathematical reasoning under minor input perturbations still lacks systematic investigation. Existing methods generally suffer from limited scalability, weak semantic preservation, and high costs. Therefore, we propose MSCR, an automated adversarial attack method based on multi-source candidate replacement. By combining three information sources including cosine similarity in the embedding space of LLMs, the WordNet dictionary, and contextual predictions from a masked language model, we generate for each word in the input question a set of semantically similar candidates, which are then filtered and substituted one by one to carry out the attack. We conduct large-scale experiments on LLMs using the GSM8K and MATH500 benchmarks. The results show that even a slight perturbation involving only a single word can significantly reduce the accuracy of all models, with the maximum drop reaching 49.89% on GSM8K and 35.40% on MATH500, while preserving the high semantic consistency of the perturbed questions. Further analysis reveals that perturbations not only lead to incorrect outputs but also substantially increase the average response length, which results in more redundant reasoning paths and higher computational resource consumption. These findings highlight the robustness deficiencies and efficiency bottlenecks of current LLMs in mathematical reasoning tasks.



## **7. Decoding Latent Attack Surfaces in LLMs: Prompt Injection via HTML in Web Summarization**

cs.CR

**SubmitDate**: 2025-11-12    [abs](http://arxiv.org/abs/2509.05831v3) [paper-pdf](None)

**Authors**: Ishaan Verma, Arsheya Yadav

**Abstract**: Large Language Models (LLMs) are increasingly integrated into web-based systems for content summarization, yet their susceptibility to prompt injection attacks remains a pressing concern. In this study, we explore how non-visible HTML elements such as <meta>, aria-label, and alt attributes can be exploited to embed adversarial instructions without altering the visible content of a webpage. We introduce a novel dataset comprising 280 static web pages, evenly divided between clean and adversarial injected versions, crafted using diverse HTML-based strategies. These pages are processed through a browser automation pipeline to extract both raw HTML and rendered text, closely mimicking real-world LLM deployment scenarios. We evaluate two state-of-the-art open-source models, Llama 4 Scout (Meta) and Gemma 9B IT (Google), on their ability to summarize this content. Using both lexical (ROUGE-L) and semantic (SBERT cosine similarity) metrics, along with manual annotations, we assess the impact of these covert injections. Our findings reveal that over 29% of injected samples led to noticeable changes in the Llama 4 Scout summaries, while Gemma 9B IT showed a lower, yet non-trivial, success rate of 15%. These results highlight a critical and largely overlooked vulnerability in LLM driven web pipelines, where hidden adversarial content can subtly manipulate model outputs. Our work offers a reproducible framework and benchmark for evaluating HTML-based prompt injection and underscores the urgent need for robust mitigation strategies in LLM applications involving web content.



## **8. Say It Differently: Linguistic Styles as Jailbreak Vectors**

cs.CL

**SubmitDate**: **NEW** 2025-11-14    [abs](http://arxiv.org/abs/2511.10519v1) [paper-pdf](None)

**Authors**: Srikant Panda, Avinash Rai

**Abstract**: Large Language Models (LLMs) are commonly evaluated for robustness against paraphrased or semantically equivalent jailbreak prompts, yet little attention has been paid to linguistic variation as an attack surface. In this work, we systematically study how linguistic styles such as fear or curiosity can reframe harmful intent and elicit unsafe responses from aligned models. We construct style-augmented jailbreak benchmark by transforming prompts from 3 standard datasets into 11 distinct linguistic styles using handcrafted templates and LLM-based rewrites, while preserving semantic intent. Evaluating 16 open- and close-source instruction-tuned models, we find that stylistic reframing increases jailbreak success rates by up to +57 percentage points. Styles such as fearful, curious and compassionate are most effective and contextualized rewrites outperform templated variants.   To mitigate this, we introduce a style neutralization preprocessing step using a secondary LLM to strip manipulative stylistic cues from user inputs, significantly reducing jailbreak success rates. Our findings reveal a systemic and scaling-resistant vulnerability overlooked in current safety pipelines.



## **9. Speech-Audio Compositional Attacks on Multimodal LLMs and Their Mitigation with SALMONN-Guard**

cs.SD

**SubmitDate**: **NEW** 2025-11-14    [abs](http://arxiv.org/abs/2511.10222v1) [paper-pdf](None)

**Authors**: Yudong Yang, Xuezhen Zhang, Zhifeng Han, Siyin Wang, Jimin Zhuang, Zengrui Jin, Jing Shao, Guangzhi Sun, Chao Zhang

**Abstract**: Recent progress in large language models (LLMs) has enabled understanding of both speech and non-speech audio, but exposing new safety risks emerging from complex audio inputs that are inadequately handled by current safeguards. We introduce SACRED-Bench (Speech-Audio Composition for RED-teaming) to evaluate the robustness of LLMs under complex audio-based attacks. Unlike existing perturbation-based methods that rely on noise optimization or white-box access, SACRED-Bench exploits speech-audio composition mechanisms. SACRED-Bench adopts three mechanisms: (a) speech overlap and multi-speaker dialogue, which embeds harmful prompts beneath or alongside benign speech; (b) speech-audio mixture, which imply unsafe intent via non-speech audio alongside benign speech or audio; and (c) diverse spoken instruction formats (open-ended QA, yes/no) that evade text-only filters. Experiments show that, even Gemini 2.5 Pro, the state-of-the-art proprietary LLM, still exhibits 66% attack success rate in SACRED-Bench test set, exposing vulnerabilities under cross-modal, speech-audio composition attacks. To bridge this gap, we propose SALMONN-Guard, a safeguard LLM that jointly inspects speech, audio, and text for safety judgments, reducing attack success down to 20%. Our results highlight the need for audio-aware defenses for the safety of multimodal LLMs. The benchmark and SALMONN-Guard checkpoints can be found at https://huggingface.co/datasets/tsinghua-ee/SACRED-Bench. Warning: this paper includes examples that may be offensive or harmful.



## **10. MTAttack: Multi-Target Backdoor Attacks against Large Vision-Language Models**

cs.CV

AAAI2026, with supplementary material

**SubmitDate**: **NEW** 2025-11-14    [abs](http://arxiv.org/abs/2511.10098v1) [paper-pdf](None)

**Authors**: Zihan Wang, Guansong Pang, Wenjun Miao, Jin Zheng, Xiao Bai

**Abstract**: Recent advances in Large Visual Language Models (LVLMs) have demonstrated impressive performance across various vision-language tasks by leveraging large-scale image-text pretraining and instruction tuning. However, the security vulnerabilities of LVLMs have become increasingly concerning, particularly their susceptibility to backdoor attacks. Existing backdoor attacks focus on single-target attacks, i.e., targeting a single malicious output associated with a specific trigger. In this work, we uncover multi-target backdoor attacks, where multiple independent triggers corresponding to different attack targets are added in a single pass of training, posing a greater threat to LVLMs in real-world applications. Executing such attacks in LVLMs is challenging since there can be many incorrect trigger-target mappings due to severe feature interference among different triggers. To address this challenge, we propose MTAttack, the first multi-target backdoor attack framework for enforcing accurate multiple trigger-target mappings in LVLMs. The core of MTAttack is a novel optimization method with two constraints, namely Proxy Space Partitioning constraint and Trigger Prototype Anchoring constraint. It jointly optimizes multiple triggers in the latent space, with each trigger independently mapping clean images to a unique proxy class while at the same time guaranteeing their separability. Experiments on popular benchmarks demonstrate a high success rate of MTAttack for multi-target attacks, substantially outperforming existing attack methods. Furthermore, our attack exhibits strong generalizability across datasets and robustness against backdoor defense strategies. These findings highlight the vulnerability of LVLMs to multi-target backdoor attacks and underscore the urgent need for mitigating such threats. Code is available at https://github.com/mala-lab/MTAttack.



## **11. Phantom Menace: Exploring and Enhancing the Robustness of VLA Models against Physical Sensor Attacks**

cs.RO

Accepted by AAAI 2026

**SubmitDate**: **NEW** 2025-11-14    [abs](http://arxiv.org/abs/2511.10008v1) [paper-pdf](None)

**Authors**: Xuancun Lu, Jiaxiang Chen, Shilin Xiao, Zizhi Jin, Zhangrui Chen, Hanwen Yu, Bohan Qian, Ruochen Zhou, Xiaoyu Ji, Wenyuan Xu

**Abstract**: Vision-Language-Action (VLA) models revolutionize robotic systems by enabling end-to-end perception-to-action pipelines that integrate multiple sensory modalities, such as visual signals processed by cameras and auditory signals captured by microphones. This multi-modality integration allows VLA models to interpret complex, real-world environments using diverse sensor data streams. Given the fact that VLA-based systems heavily rely on the sensory input, the security of VLA models against physical-world sensor attacks remains critically underexplored.   To address this gap, we present the first systematic study of physical sensor attacks against VLAs, quantifying the influence of sensor attacks and investigating the defenses for VLA models. We introduce a novel ``Real-Sim-Real'' framework that automatically simulates physics-based sensor attack vectors, including six attacks targeting cameras and two targeting microphones, and validates them on real robotic systems. Through large-scale evaluations across various VLA architectures and tasks under varying attack parameters, we demonstrate significant vulnerabilities, with susceptibility patterns that reveal critical dependencies on task types and model designs. We further develop an adversarial-training-based defense that enhances VLA robustness against out-of-distribution physical perturbations caused by sensor attacks while preserving model performance. Our findings expose an urgent need for standardized robustness benchmarks and mitigation strategies to secure VLA deployments in safety-critical environments.



## **12. EnchTable: Unified Safety Alignment Transfer in Fine-tuned Large Language Models**

cs.CL

Accepted by IEEE Symposium on Security and Privacy (S&P) 2026

**SubmitDate**: **NEW** 2025-11-14    [abs](http://arxiv.org/abs/2511.09880v1) [paper-pdf](None)

**Authors**: Jialin Wu, Kecen Li, Zhicong Huang, Xinfeng Li, Xiaofeng Wang, Cheng Hong

**Abstract**: Many machine learning models are fine-tuned from large language models (LLMs) to achieve high performance in specialized domains like code generation, biomedical analysis, and mathematical problem solving. However, this fine-tuning process often introduces a critical vulnerability: the systematic degradation of safety alignment, undermining ethical guidelines and increasing the risk of harmful outputs. Addressing this challenge, we introduce EnchTable, a novel framework designed to transfer and maintain safety alignment in downstream LLMs without requiring extensive retraining. EnchTable leverages a Neural Tangent Kernel (NTK)-based safety vector distillation method to decouple safety constraints from task-specific reasoning, ensuring compatibility across diverse model architectures and sizes. Additionally, our interference-aware merging technique effectively balances safety and utility, minimizing performance compromises across various task domains. We implemented a fully functional prototype of EnchTable on three different task domains and three distinct LLM architectures, and evaluated its performance through extensive experiments on eleven diverse datasets, assessing both utility and model safety. Our evaluations include LLMs from different vendors, demonstrating EnchTable's generalization capability. Furthermore, EnchTable exhibits robust resistance to static and dynamic jailbreaking attacks, outperforming vendor-released safety models in mitigating adversarial prompts. Comparative analyses with six parameter modification methods and two inference-time alignment baselines reveal that EnchTable achieves a significantly lower unsafe rate, higher utility score, and universal applicability across different task domains. Additionally, we validate EnchTable can be seamlessly integrated into various deployment pipelines without significant overhead.



## **13. Adaptive and Robust Data Poisoning Detection and Sanitization in Wearable IoT Systems using Large Language Models**

cs.LG

**SubmitDate**: 2025-11-11    [abs](http://arxiv.org/abs/2511.02894v2) [paper-pdf](None)

**Authors**: W. K. M Mithsara, Ning Yang, Ahmed Imteaj, Hussein Zangoti, Abdur R. Shahid

**Abstract**: The widespread integration of wearable sensing devices in Internet of Things (IoT) ecosystems, particularly in healthcare, smart homes, and industrial applications, has required robust human activity recognition (HAR) techniques to improve functionality and user experience. Although machine learning models have advanced HAR, they are increasingly susceptible to data poisoning attacks that compromise the data integrity and reliability of these systems. Conventional approaches to defending against such attacks often require extensive task-specific training with large, labeled datasets, which limits adaptability in dynamic IoT environments. This work proposes a novel framework that uses large language models (LLMs) to perform poisoning detection and sanitization in HAR systems, utilizing zero-shot, one-shot, and few-shot learning paradigms. Our approach incorporates \textit{role play} prompting, whereby the LLM assumes the role of expert to contextualize and evaluate sensor anomalies, and \textit{think step-by-step} reasoning, guiding the LLM to infer poisoning indicators in the raw sensor data and plausible clean alternatives. These strategies minimize reliance on curation of extensive datasets and enable robust, adaptable defense mechanisms in real-time. We perform an extensive evaluation of the framework, quantifying detection accuracy, sanitization quality, latency, and communication cost, thus demonstrating the practicality and effectiveness of LLMs in improving the security and reliability of wearable IoT systems.



## **14. E2E-VGuard: Adversarial Prevention for Production LLM-based End-To-End Speech Synthesis**

cs.SD

Accepted to NeurIPS 2025

**SubmitDate**: 2025-11-11    [abs](http://arxiv.org/abs/2511.07099v1) [paper-pdf](None)

**Authors**: Zhisheng Zhang, Derui Wang, Yifan Mi, Zhiyong Wu, Jie Gao, Yuxin Cao, Kai Ye, Minhui Xue, Jie Hao

**Abstract**: Recent advancements in speech synthesis technology have enriched our daily lives, with high-quality and human-like audio widely adopted across real-world applications. However, malicious exploitation like voice-cloning fraud poses severe security risks. Existing defense techniques struggle to address the production large language model (LLM)-based speech synthesis. While previous studies have considered the protection for fine-tuning synthesizers, they assume manually annotated transcripts. Given the labor intensity of manual annotation, end-to-end (E2E) systems leveraging automatic speech recognition (ASR) to generate transcripts are becoming increasingly prevalent, e.g., voice cloning via commercial APIs. Therefore, this E2E speech synthesis also requires new security mechanisms. To tackle these challenges, we propose E2E-VGuard, a proactive defense framework for two emerging threats: (1) production LLM-based speech synthesis, and (2) the novel attack arising from ASR-driven E2E scenarios. Specifically, we employ the encoder ensemble with a feature extractor to protect timbre, while ASR-targeted adversarial examples disrupt pronunciation. Moreover, we incorporate the psychoacoustic model to ensure perturbative imperceptibility. For a comprehensive evaluation, we test 16 open-source synthesizers and 3 commercial APIs across Chinese and English datasets, confirming E2E-VGuard's effectiveness in timbre and pronunciation protection. Real-world deployment validation is also conducted. Our code and demo page are available at https://wxzyd123.github.io/e2e-vguard/.



## **15. From Pretrain to Pain: Adversarial Vulnerability of Video Foundation Models Without Task Knowledge**

cs.CV

AAAI 2026 (Oral presentation)

**SubmitDate**: 2025-11-11    [abs](http://arxiv.org/abs/2511.07049v1) [paper-pdf](None)

**Authors**: Hui Lu, Yi Yu, Song Xia, Yiming Yang, Deepu Rajan, Boon Poh Ng, Alex Kot, Xudong Jiang

**Abstract**: Large-scale Video Foundation Models (VFMs) has significantly advanced various video-related tasks, either through task-specific models or Multi-modal Large Language Models (MLLMs). However, the open accessibility of VFMs also introduces critical security risks, as adversaries can exploit full knowledge of the VFMs to launch potent attacks. This paper investigates a novel and practical adversarial threat scenario: attacking downstream models or MLLMs fine-tuned from open-source VFMs, without requiring access to the victim task, training data, model query, and architecture. In contrast to conventional transfer-based attacks that rely on task-aligned surrogate models, we demonstrate that adversarial vulnerabilities can be exploited directly from the VFMs. To this end, we propose the Transferable Video Attack (TVA), a temporal-aware adversarial attack method that leverages the temporal representation dynamics of VFMs to craft effective perturbations. TVA integrates a bidirectional contrastive learning mechanism to maximize the discrepancy between the clean and adversarial features, and introduces a temporal consistency loss that exploits motion cues to enhance the sequential impact of perturbations. TVA avoids the need to train expensive surrogate models or access to domain-specific data, thereby offering a more practical and efficient attack strategy. Extensive experiments across 24 video-related tasks demonstrate the efficacy of TVA against downstream models and MLLMs, revealing a previously underexplored security vulnerability in the deployment of video models.



## **16. Graph Representation-based Model Poisoning on the Heterogeneous Internet of Agents**

cs.NI

6 pages, 6 figures

**SubmitDate**: 2025-11-11    [abs](http://arxiv.org/abs/2511.07176v1) [paper-pdf](None)

**Authors**: Hanlin Cai, Houtianfu Wang, Haofan Dong, Kai Li, Ozgur B. Akan

**Abstract**: Internet of Agents (IoA) envisions a unified, agent-centric paradigm where heterogeneous large language model (LLM) agents can interconnect and collaborate at scale. Within this paradigm, federated learning (FL) serves as a key enabler that allows distributed LLM agents to co-train global models without centralizing data. However, the FL-enabled IoA system remains vulnerable to model poisoning attacks, and the prevailing distance and similarity-based defenses become fragile at billion-parameter scale and under heterogeneous data distributions. This paper proposes a graph representation-based model poisoning (GRMP) attack, which passively exploits observed benign local models to construct a parameter correlation graph and extends an adversarial variational graph autoencoder to capture and reshape higher-order dependencies. The GRMP attack synthesizes malicious local models that preserve benign-like statistics while embedding adversarial objectives, remaining elusive to detection at the server. Experiments demonstrate a gradual drop in system accuracy under the proposed attack and the ineffectiveness of the prevailing defense mechanism in detecting the attack, underscoring a severe threat to the ambitious IoA paradigm.



## **17. DP-Fusion: Token-Level Differentially Private Inference for Large Language Models**

cs.CL

Our code and data are publicly available here: https://github.com/MBZUAI-Trustworthy-ML/DP-Fusion-DPI

**SubmitDate**: 2025-11-11    [abs](http://arxiv.org/abs/2507.04531v3) [paper-pdf](None)

**Authors**: Rushil Thareja, Preslav Nakov, Praneeth Vepakomma, Nils Lukas

**Abstract**: Large language models (LLMs) do not preserve privacy at inference-time. The LLM's outputs can inadvertently reveal information about the model's context, which presents a privacy challenge when the LLM is augmented via tools or databases containing sensitive information. Existing privacy-preserving methods at inference-time have significant limitations since they (i) lack provable guarantees or (ii) have a poor utility/privacy trade-off. We propose DP-Fusion, a Differentially Private Inference (DPI) mechanism for LLMs that provably bounds the influence a set of tokens in the context can have on the LLM's output. DP-Fusion works as follows: (1) label a subset of sensitive tokens, (2) infer the LLM without any sensitive tokens to obtain a baseline, (3) infer the LLM with the sensitive tokens, and (4) blend distributions so that the final output remains within a bounded distance of the baseline distribution. While this per-token influence bound also mitigates jailbreak-style prompt injection, we focus on \emph{document privatization}, where the goal is to paraphrase a document containing sensitive tokens, e.g., personally identifiable information, so that no attacker can reliably infer them from the paraphrased document while preserving high text quality. The privacy/utility trade-off is controlled by $ε$, where $ε=0$ hides sensitive tokens entirely, while higher values trade off privacy for improved text quality. We show that our method creates token-level provably privatized documents with substantially improved theoretical and empirical privacy, achieving $6\times$ lower perplexity than related DPI methods.



## **18. AutoAdv: Automated Adversarial Prompting for Multi-Turn Jailbreaking of Large Language Models**

cs.CL

Accepted to NeurIPS 2025 Lock-LLM Workshop. Code is available at https://github.com/AAN-AutoAdv/AutoAdv

**SubmitDate**: 2025-11-11    [abs](http://arxiv.org/abs/2511.02376v2) [paper-pdf](None)

**Authors**: Aashray Reddy, Andrew Zagula, Nicholas Saban, Kevin Zhu

**Abstract**: Large Language Models (LLMs) remain vulnerable to jailbreaking attacks where adversarial prompts elicit harmful outputs, yet most evaluations focus on single-turn interactions while real-world attacks unfold through adaptive multi-turn conversations. We present AutoAdv, a training-free framework for automated multi-turn jailbreaking that achieves up to 95% attack success rate on Llama-3.1-8B within six turns a 24 percent improvement over single turn baselines. AutoAdv uniquely combines three adaptive mechanisms: a pattern manager that learns from successful attacks to enhance future prompts, a temperature manager that dynamically adjusts sampling parameters based on failure modes, and a two-phase rewriting strategy that disguises harmful requests then iteratively refines them. Extensive evaluation across commercial and open-source models (GPT-4o-mini, Qwen3-235B, Mistral-7B) reveals persistent vulnerabilities in current safety mechanisms, with multi-turn attacks consistently outperforming single-turn approaches. These findings demonstrate that alignment strategies optimized for single-turn interactions fail to maintain robustness across extended conversations, highlighting an urgent need for multi-turn-aware defenses.



## **19. When AI Meets the Web: Prompt Injection Risks in Third-Party AI Chatbot Plugins**

cs.CR

At IEEE S&P 2026

**SubmitDate**: 2025-11-11    [abs](http://arxiv.org/abs/2511.05797v1) [paper-pdf](None)

**Authors**: Yigitcan Kaya, Anton Landerer, Stijn Pletinckx, Michelle Zimmermann, Christopher Kruegel, Giovanni Vigna

**Abstract**: Prompt injection attacks pose a critical threat to large language models (LLMs), with prior work focusing on cutting-edge LLM applications like personal copilots. In contrast, simpler LLM applications, such as customer service chatbots, are widespread on the web, yet their security posture and exposure to such attacks remain poorly understood. These applications often rely on third-party chatbot plugins that act as intermediaries to commercial LLM APIs, offering non-expert website builders intuitive ways to customize chatbot behaviors. To bridge this gap, we present the first large-scale study of 17 third-party chatbot plugins used by over 10,000 public websites, uncovering previously unknown prompt injection risks in practice. First, 8 of these plugins (used by 8,000 websites) fail to enforce the integrity of the conversation history transmitted in network requests between the website visitor and the chatbot. This oversight amplifies the impact of direct prompt injection attacks by allowing adversaries to forge conversation histories (including fake system messages), boosting their ability to elicit unintended behavior (e.g., code generation) by 3 to 8x. Second, 15 plugins offer tools, such as web-scraping, to enrich the chatbot's context with website-specific content. However, these tools do not distinguish the website's trusted content (e.g., product descriptions) from untrusted, third-party content (e.g., customer reviews), introducing a risk of indirect prompt injection. Notably, we found that ~13% of e-commerce websites have already exposed their chatbots to third-party content. We systematically evaluate both vulnerabilities through controlled experiments grounded in real-world observations, focusing on factors such as system prompt design and the underlying LLM. Our findings show that many plugins adopt insecure practices that undermine the built-in LLM safeguards.



## **20. A Self-Improving Architecture for Dynamic Safety in Large Language Models**

cs.SE

Under review at the journal Information and Software Technology (Special Issue on Software Architecture for AI-Driven Systems)

**SubmitDate**: 2025-11-12    [abs](http://arxiv.org/abs/2511.07645v1) [paper-pdf](None)

**Authors**: Tyler Slater

**Abstract**: Context: The integration of Large Language Models (LLMs) into core software systems is accelerating. However, existing software architecture patterns are static, while current safety assurance methods are not scalable, leaving systems vulnerable to novel adversarial threats.   Objective: To design, implement, and evaluate a novel software architecture that enables an AI-driven system to autonomously and continuously adapt its own safety protocols at runtime.   Method: We propose the Self-Improving Safety Framework (SISF), a runtime architecture that couples an unprotected, unaligned base LLM (mistralai/Mistral-7B-v0.1) with a dynamic feedback loop. This loop consists of an AI Adjudicator (GPT-4o) for breach detection and a Policy Synthesis Module (GPT-4 Turbo) that autonomously generates new, generalized safety policies (both heuristic and semantic) in response to failures.   Results: We conducted a dynamic learning evaluation using the 520-prompt AdvBench dataset. The unprotected model was 100% vulnerable. Our SISF, starting from zero policies, demonstrated a clear learning curve: it detected 237 breaches, autonomously synthesized 234 new policies, and reduced the overall Attack Success Rate (ASR) to 45.58%. In a subsequent test on 520 benign prompts, the SISF achieved a 0.00% False Positive Rate (FPR), proving its ability to adapt without compromising user utility.   Conclusion: An architectural approach to AI safety, based on the principles of self-adaptation, is a viable and effective strategy. Our framework demonstrates a practical path towards building more robust, resilient, and scalable AI-driven systems, shifting safety assurance from a static, pre-deployment activity to an automated, runtime process.



## **21. Comparing Reconstruction Attacks on Pretrained Versus Full Fine-tuned Large Language Model Embeddings on Homo Sapiens Splice Sites Genomic Data**

cs.LG

**SubmitDate**: 2025-11-12    [abs](http://arxiv.org/abs/2511.07481v1) [paper-pdf](None)

**Authors**: Reem Al-Saidi, Erman Ayday, Ziad Kobti

**Abstract**: This study investigates embedding reconstruction attacks in large language models (LLMs) applied to genomic sequences, with a specific focus on how fine-tuning affects vulnerability to these attacks. Building upon Pan et al.'s seminal work demonstrating that embeddings from pretrained language models can leak sensitive information, we conduct a comprehensive analysis using the HS3D genomic dataset to determine whether task-specific optimization strengthens or weakens privacy protections. Our research extends Pan et al.'s work in three significant dimensions. First, we apply their reconstruction attack pipeline to pretrained and fine-tuned model embeddings, addressing a critical gap in their methodology that did not specify embedding types. Second, we implement specialized tokenization mechanisms tailored specifically for DNA sequences, enhancing the model's ability to process genomic data, as these models are pretrained on natural language and not DNA. Third, we perform a detailed comparative analysis examining position-specific, nucleotide-type, and privacy changes between pretrained and fine-tuned embeddings. We assess embeddings vulnerabilities across different types and dimensions, providing deeper insights into how task adaptation shifts privacy risks throughout genomic sequences. Our findings show a clear distinction in reconstruction vulnerability between pretrained and fine-tuned embeddings. Notably, fine-tuning strengthens resistance to reconstruction attacks in multiple architectures -- XLNet (+19.8\%), GPT-2 (+9.8\%), and BERT (+7.8\%) -- pointing to task-specific optimization as a potential privacy enhancement mechanism. These results highlight the need for advanced protective mechanisms for language models processing sensitive genomic data, while highlighting fine-tuning as a potential privacy-enhancing technique worth further exploration.



## **22. Reasoning Up the Instruction Ladder for Controllable Language Models**

cs.CL

**SubmitDate**: 2025-11-13    [abs](http://arxiv.org/abs/2511.04694v2) [paper-pdf](None)

**Authors**: Zishuo Zheng, Vidhisha Balachandran, Chan Young Park, Faeze Brahman, Sachin Kumar

**Abstract**: As large language model (LLM) based systems take on high-stakes roles in real-world decision-making, they must reconcile competing instructions from multiple sources (e.g., model developers, users, and tools) within a single prompt context. Thus, enforcing an instruction hierarchy (IH) in LLMs, where higher-level directives override lower-priority requests, is critical for the reliability and controllability of LLMs. In this work, we reframe instruction hierarchy resolution as a reasoning task. Specifically, the model must first "think" about the relationship between a given user prompt and higher-priority (system) instructions before generating a response. To enable this capability via training, we construct VerIH, an instruction hierarchy dataset of constraint-following tasks with verifiable answers. This dataset comprises both aligned and conflicting system-user instructions. We show that lightweight reinforcement learning with VerIH effectively transfers general reasoning capabilities of models to instruction prioritization. Our finetuned models achieve consistent improvements on instruction following and instruction hierarchy benchmarks. This reasoning ability also generalizes to safety-critical settings beyond the training distribution. By treating safety issues as resolving conflicts between adversarial user inputs and predefined higher-priority policies, our trained model enhances robustness against jailbreak and prompt injection attacks. These results demonstrate that reasoning over instruction hierarchies provides a practical path to reliable LLMs, where updates to system prompts yield controllable and robust changes in model behavior.



## **23. SecInfer: Preventing Prompt Injection via Inference-time Scaling**

cs.CR

**SubmitDate**: 2025-11-13    [abs](http://arxiv.org/abs/2509.24967v3) [paper-pdf](None)

**Authors**: Yupei Liu, Yanting Wang, Yuqi Jia, Jinyuan Jia, Neil Zhenqiang Gong

**Abstract**: Prompt injection attacks pose a pervasive threat to the security of Large Language Models (LLMs). State-of-the-art prevention-based defenses typically rely on fine-tuning an LLM to enhance its security, but they achieve limited effectiveness against strong attacks. In this work, we propose \emph{SecInfer}, a novel defense against prompt injection attacks built on \emph{inference-time scaling}, an emerging paradigm that boosts LLM capability by allocating more compute resources for reasoning during inference. SecInfer consists of two key steps: \emph{system-prompt-guided sampling}, which generates multiple responses for a given input by exploring diverse reasoning paths through a varied set of system prompts, and \emph{target-task-guided aggregation}, which selects the response most likely to accomplish the intended task. Extensive experiments show that, by leveraging additional compute at inference, SecInfer effectively mitigates both existing and adaptive prompt injection attacks, outperforming state-of-the-art defenses as well as existing inference-time scaling approaches.



## **24. Backdoor Attacks Against Speech Language Models**

cs.CL

**SubmitDate**: **NEW** 2025-11-14    [abs](http://arxiv.org/abs/2510.01157v2) [paper-pdf](None)

**Authors**: Alexandrine Fortier, Thomas Thebaud, Jesús Villalba, Najim Dehak, Patrick Cardinal

**Abstract**: Large Language Models (LLMs) and their multimodal extensions are becoming increasingly popular. One common approach to enable multimodality is to cascade domain-specific encoders with an LLM, making the resulting model inherit vulnerabilities from all of its components. In this work, we present the first systematic study of audio backdoor attacks against speech language models. We demonstrate its effectiveness across four speech encoders and three datasets, covering four tasks: automatic speech recognition (ASR), speech emotion recognition, and gender and age prediction. The attack consistently achieves high success rates, ranging from 90.76% to 99.41%. To better understand how backdoors propagate, we conduct a component-wise analysis to identify the most vulnerable stages of the pipeline. Finally, we propose a fine-tuning-based defense that mitigates the threat of poisoned pretrained encoders.



## **25. EduGuardBench: A Holistic Benchmark for Evaluating the Pedagogical Fidelity and Adversarial Safety of LLMs as Simulated Teachers**

cs.CL

22 pages, 9 figures, accepted by AAAI2026 as oral paper

**SubmitDate**: 2025-11-11    [abs](http://arxiv.org/abs/2511.06890v1) [paper-pdf](None)

**Authors**: Yilin Jiang, Mingzi Zhang, Xuanyu Yin, Sheng Jin, Suyu Lu, Zuocan Ying, Zengyi Yu, Xiangjie Kong

**Abstract**: Large Language Models for Simulating Professions (SP-LLMs), particularly as teachers, are pivotal for personalized education. However, ensuring their professional competence and ethical safety is a critical challenge, as existing benchmarks fail to measure role-playing fidelity or address the unique teaching harms inherent in educational scenarios. To address this, we propose EduGuardBench, a dual-component benchmark. It assesses professional fidelity using a Role-playing Fidelity Score (RFS) while diagnosing harms specific to the teaching profession. It also probes safety vulnerabilities using persona-based adversarial prompts targeting both general harms and, particularly, academic misconduct, evaluated with metrics including Attack Success Rate (ASR) and a three-tier Refusal Quality assessment. Our extensive experiments on 14 leading models reveal a stark polarization in performance. While reasoning-oriented models generally show superior fidelity, incompetence remains the dominant failure mode across most models. The adversarial tests uncovered a counterintuitive scaling paradox, where mid-sized models can be the most vulnerable, challenging monotonic safety assumptions. Critically, we identified a powerful Educational Transformation Effect: the safest models excel at converting harmful requests into teachable moments by providing ideal Educational Refusals. This capacity is strongly negatively correlated with ASR, revealing a new dimension of advanced AI safety. EduGuardBench thus provides a reproducible framework that moves beyond siloed knowledge tests toward a holistic assessment of professional, ethical, and pedagogical alignment, uncovering complex dynamics essential for deploying trustworthy AI in education. See https://github.com/YL1N/EduGuardBench for Materials.



## **26. SAFENLIDB: A Privacy-Preserving Safety Alignment Framework for LLM-based Natural Language Database Interfaces**

cs.CL

AAAI 2026 Extended Version

**SubmitDate**: 2025-11-12    [abs](http://arxiv.org/abs/2511.06778v2) [paper-pdf](None)

**Authors**: Ruiheng Liu, XiaoBing Chen, Jinyu Zhang, Qiongwen Zhang, Yu Zhang, Bailong Yang

**Abstract**: The rapid advancement of Large Language Models (LLMs) has driven significant progress in Natural Language Interface to Database (NLIDB). However, the widespread adoption of LLMs has raised critical privacy and security concerns. During interactions, LLMs may unintentionally expose confidential database contents or be manipulated by attackers to exfiltrate data through seemingly benign queries. While current efforts typically rely on rule-based heuristics or LLM agents to mitigate this leakage risk, these methods still struggle with complex inference-based attacks, suffer from high false positive rates, and often compromise the reliability of SQL queries. To address these challenges, we propose \textsc{SafeNlidb}, a novel privacy-security alignment framework for LLM-based NLIDB. The framework features an automated pipeline that generates hybrid chain-of-thought interaction data from scratch, seamlessly combining implicit security reasoning with SQL generation. Additionally, we introduce reasoning warm-up and alternating preference optimization to overcome the multi-preference oscillations of Direct Preference Optimization (DPO), enabling LLMs to produce security-aware SQL through fine-grained reasoning without the need for human-annotated preference data. Extensive experiments demonstrate that our method outperforms both larger-scale LLMs and ideal-setting baselines, achieving significant security improvements while preserving high utility. WARNING: This work may contain content that is offensive and harmful!



## **27. CoSPED: Consistent Soft Prompt Targeted Data Extraction and Defense**

cs.CR

**SubmitDate**: 2025-11-12    [abs](http://arxiv.org/abs/2510.11137v2) [paper-pdf](None)

**Authors**: Zhuochen Yang, Kar Wai Fok, Vrizlynn L. L. Thing

**Abstract**: Large language models have gained widespread attention recently, but their potential security vulnerabilities, especially privacy leakage, are also becoming apparent. To test and evaluate for data extraction risks in LLM, we proposed CoSPED, short for Consistent Soft Prompt targeted data Extraction and Defense. We introduce several innovative components, including Dynamic Loss, Additive Loss, Common Loss, and Self Consistency Decoding Strategy, and tested to enhance the consistency of the soft prompt tuning process. Through extensive experimentation with various combinations, we achieved an extraction rate of 65.2% at a 50-token prefix comparison. Our comparisons of CoSPED with other reference works confirm our superior extraction rates. We evaluate CoSPED on more scenarios, achieving Pythia model extraction rate of 51.7% and introducing cross-model comparison. Finally, we explore defense through Rank-One Model Editing and achieve a reduction in the extraction rate to 1.6%, which proves that our analysis of extraction mechanisms can directly inform effective mitigation strategies against soft prompt-based attacks.



## **28. CyberSOCEval: Benchmarking LLMs Capabilities for Malware Analysis and Threat Intelligence Reasoning**

cs.CR

**SubmitDate**: 2025-11-12    [abs](http://arxiv.org/abs/2509.20166v2) [paper-pdf](None)

**Authors**: Lauren Deason, Adam Bali, Ciprian Bejean, Diana Bolocan, James Crnkovich, Ioana Croitoru, Krishna Durai, Chase Midler, Calin Miron, David Molnar, Brad Moon, Bruno Ostarcevic, Alberto Peltea, Matt Rosenberg, Catalin Sandu, Arthur Saputkin, Sagar Shah, Daniel Stan, Ernest Szocs, Shengye Wan, Spencer Whitman, Sven Krasser, Joshua Saxe

**Abstract**: Today's cyber defenders are overwhelmed by a deluge of security alerts, threat intelligence signals, and shifting business context, creating an urgent need for AI systems to enhance operational security work. While Large Language Models (LLMs) have the potential to automate and scale Security Operations Center (SOC) operations, existing evaluations do not fully assess the scenarios most relevant to real-world defenders. This lack of informed evaluation impacts both AI developers and those applying LLMs to SOC automation. Without clear insight into LLM performance in real-world security scenarios, developers lack a north star for development, and users cannot reliably select the most effective models. Meanwhile, malicious actors are using AI to scale cyber attacks, highlighting the need for open source benchmarks to drive adoption and community-driven improvement among defenders and model developers. To address this, we introduce CyberSOCEval, a new suite of open source benchmarks within CyberSecEval 4. CyberSOCEval includes benchmarks tailored to evaluate LLMs in two tasks: Malware Analysis and Threat Intelligence Reasoning--core defensive domains with inadequate coverage in current benchmarks. Our evaluations show that larger, more modern LLMs tend to perform better, confirming the training scaling laws paradigm. We also find that reasoning models leveraging test time scaling do not achieve the same boost as in coding and math, suggesting these models have not been trained to reason about cybersecurity analysis, and pointing to a key opportunity for improvement. Finally, current LLMs are far from saturating our evaluations, showing that CyberSOCEval presents a significant challenge for AI developers to improve cyber defense capabilities.



## **29. Cost-Minimized Label-Flipping Poisoning Attack to LLM Alignment**

cs.LG

accepted for AAAI 2026 Special Track on AI Alignment

**SubmitDate**: 2025-11-13    [abs](http://arxiv.org/abs/2511.09105v1) [paper-pdf](None)

**Authors**: Shigeki Kusaka, Keita Saito, Mikoto Kudo, Takumi Tanabe, Akifumi Wachi, Youhei Akimoto

**Abstract**: Large language models (LLMs) are increasingly deployed in real-world systems, making it critical to understand their vulnerabilities. While data poisoning attacks during RLHF/DPO alignment have been studied empirically, their theoretical foundations remain unclear. We investigate the minimum-cost poisoning attack required to steer an LLM's policy toward an attacker's target by flipping preference labels during RLHF/DPO, without altering the compared outputs. We formulate this as a convex optimization problem with linear constraints, deriving lower and upper bounds on the minimum attack cost. As a byproduct of this theoretical analysis, we show that any existing label-flipping attack can be post-processed via our proposed method to reduce the number of label flips required while preserving the intended poisoning effect. Empirical results demonstrate that this cost-minimization post-processing can significantly reduce poisoning costs over baselines, particularly when the reward model's feature dimension is small relative to the dataset size. These findings highlight fundamental vulnerabilities in RLHF/DPO pipelines and provide tools to evaluate their robustness against low-cost poisoning attacks.



## **30. Joint-GCG: Unified Gradient-Based Poisoning Attacks on Retrieval-Augmented Generation Systems**

cs.CR

**SubmitDate**: 2025-11-13    [abs](http://arxiv.org/abs/2506.06151v2) [paper-pdf](None)

**Authors**: Haowei Wang, Rupeng Zhang, Junjie Wang, Mingyang Li, Yuekai Huang, Dandan Wang, Qing Wang

**Abstract**: Retrieval-Augmented Generation (RAG) systems enhance Large Language Models (LLMs) by retrieving relevant documents from external corpora before generating responses. This approach significantly expands LLM capabilities by leveraging vast, up-to-date external knowledge. However, this reliance on external knowledge makes RAG systems vulnerable to corpus poisoning attacks that manipulate generated outputs via poisoned document injection. Existing poisoning attack strategies typically treat the retrieval and generation stages as disjointed, limiting their effectiveness. We propose Joint-GCG, the first framework to unify gradient-based attacks across both retriever and generator models through three innovations: (1) Cross-Vocabulary Projection for aligning embedding spaces, (2) Gradient Tokenization Alignment for synchronizing token-level gradient signals, and (3) Adaptive Weighted Fusion for dynamically balancing attacking objectives. Evaluations demonstrate that Joint-GCG achieves at most 25% and an average of 5% higher attack success rate than previous methods across multiple retrievers and generators. While optimized under a white-box assumption, the generated poisons show unprecedented transferability to unseen models. Joint-GCG's innovative unification of gradient-based attacks across retrieval and generation stages fundamentally reshapes our understanding of vulnerabilities within RAG systems. Our code is available at https://github.com/NicerWang/Joint-GCG.



## **31. Siren: A Learning-Based Multi-Turn Attack Framework for Simulating Real-World Human Jailbreak Behaviors**

cs.CL

Accepted at ACSAC 2025

**SubmitDate**: **NEW** 2025-11-14    [abs](http://arxiv.org/abs/2501.14250v2) [paper-pdf](None)

**Authors**: Yi Zhao, Youzhi Zhang

**Abstract**: Large language models (LLMs) are widely used in real-world applications, raising concerns about their safety and trustworthiness. While red-teaming with jailbreak prompts exposes the vulnerabilities of LLMs, current efforts focus primarily on single-turn attacks, overlooking the multi-turn strategies used by real-world adversaries. Existing multi-turn methods rely on static patterns or predefined logical chains, failing to account for the dynamic strategies during attacks. We propose Siren, a learning-based multi-turn attack framework designed to simulate real-world human jailbreak behaviors. Siren consists of three stages: (1) MiniMax-driven training set construction utilizing Turn-Level LLM feedback, (2) post-training attackers with supervised fine-tuning (SFT) and direct preference optimization (DPO), and (3) interactions between the attacking and target LLMs. Experiments demonstrate that Siren achieves an attack success rate (ASR) of 90% with LLaMA-3-8B as the attacker against Gemini-1.5-Pro as the target model, and 70% with Mistral-7B against GPT-4o, significantly outperforming single-turn baselines. Moreover, Siren with a 7B-scale model achieves performance comparable to a multi-turn baseline that leverages GPT-4o as the attacker, while requiring fewer turns and employing decomposition strategies that are better semantically aligned with attack goals. We hope Siren inspires the development of stronger defenses against advanced multi-turn jailbreak attacks under realistic scenarios. Code is available at https://github.com/YiyiyiZhao/siren. Warning: This paper contains potentially harmful text.



## **32. Secure Retrieval-Augmented Generation against Poisoning Attacks**

cs.CR

To appear in IEEE BigData 2025

**SubmitDate**: 2025-11-11    [abs](http://arxiv.org/abs/2510.25025v2) [paper-pdf](None)

**Authors**: Zirui Cheng, Jikai Sun, Anjun Gao, Yueyang Quan, Zhuqing Liu, Xiaohua Hu, Minghong Fang

**Abstract**: Large language models (LLMs) have transformed natural language processing (NLP), enabling applications from content generation to decision support. Retrieval-Augmented Generation (RAG) improves LLMs by incorporating external knowledge but also introduces security risks, particularly from data poisoning, where the attacker injects poisoned texts into the knowledge database to manipulate system outputs. While various defenses have been proposed, they often struggle against advanced attacks. To address this, we introduce RAGuard, a detection framework designed to identify poisoned texts. RAGuard first expands the retrieval scope to increase the proportion of clean texts, reducing the likelihood of retrieving poisoned content. It then applies chunk-wise perplexity filtering to detect abnormal variations and text similarity filtering to flag highly similar texts. This non-parametric approach enhances RAG security, and experiments on large-scale datasets demonstrate its effectiveness in detecting and mitigating poisoning attacks, including strong adaptive attacks.



## **33. HumorReject: Decoupling LLM Safety from Refusal Prefix via A Little Humor**

cs.LG

**SubmitDate**: 2025-11-11    [abs](http://arxiv.org/abs/2501.13677v3) [paper-pdf](None)

**Authors**: Zihui Wu, Haichang Gao, Jiacheng Luo, Zhaoxiang Liu

**Abstract**: Large Language Models (LLMs) commonly rely on explicit refusal prefixes for safety, making them vulnerable to prefix injection attacks. We introduce HumorReject, a novel data-driven approach that reimagines LLM safety by decoupling it from refusal prefixes through humor as an indirect refusal strategy. Rather than explicitly rejecting harmful instructions, HumorReject responds with contextually appropriate humor that naturally defuses potentially dangerous requests. Our approach effectively addresses common "over-defense" issues while demonstrating superior robustness against various attack vectors. Our findings suggest that improvements in training data design can be as important as the alignment algorithm itself in achieving effective LLM safety. The code and dataset are available at https://github.com/wooozihui/HumorReject.



## **34. MENTOR: A Metacognition-Driven Self-Evolution Framework for Uncovering and Mitigating Implicit Risks in LLMs on Domain Tasks**

cs.AI

**SubmitDate**: 2025-11-11    [abs](http://arxiv.org/abs/2511.07107v1) [paper-pdf](None)

**Authors**: Liang Shan, Kaicheng Shen, Wen Wu, Zhenyu Ying, Chaochao Lu, Guangze Ye, Liang He

**Abstract**: Ensuring the safety and value alignment of large language models (LLMs) is critical for their deployment. Current alignment efforts primarily target explicit risks such as bias, hate speech, and violence. However, they often fail to address deeper, domain-specific implicit risks and lack a flexible, generalizable framework applicable across diverse specialized fields. Hence, we proposed MENTOR: A MEtacognition-driveN self-evoluTion framework for uncOvering and mitigating implicit Risks in LLMs on Domain Tasks. To address the limitations of labor-intensive human evaluation, we introduce a novel metacognitive self-assessment tool. This enables LLMs to reflect on potential value misalignments in their responses using strategies like perspective-taking and consequential thinking. We also release a supporting dataset of 9,000 risk queries spanning education, finance, and management to enhance domain-specific risk identification. Subsequently, based on the outcomes of metacognitive reflection, the framework dynamically generates supplementary rule knowledge graphs that extend predefined static rule trees. This enables models to actively apply validated rules to future similar challenges, establishing a continuous self-evolution cycle that enhances generalization by reducing maintenance costs and inflexibility of static systems. Finally, we employ activation steering during inference to guide LLMs in following the rules, a cost-effective method to robustly enhance enforcement across diverse contexts. Experimental results show MENTOR's effectiveness: In defensive testing across three vertical domains, the framework substantially reduces semantic attack success rates, enabling a new level of implicit risk mitigation for LLMs. Furthermore, metacognitive assessment not only aligns closely with baseline human evaluators but also delivers more thorough and insightful analysis of LLMs value alignment.



## **35. KG-DF: A Black-box Defense Framework against Jailbreak Attacks Based on Knowledge Graphs**

cs.CR

**SubmitDate**: 2025-11-12    [abs](http://arxiv.org/abs/2511.07480v1) [paper-pdf](None)

**Authors**: Shuyuan Liu, Jiawei Chen, Xiao Yang, Hang Su, Zhaoxia Yin

**Abstract**: With the widespread application of large language models (LLMs) in various fields, the security challenges they face have become increasingly prominent, especially the issue of jailbreak. These attacks induce the model to generate erroneous or uncontrolled outputs through crafted inputs, threatening the generality and security of the model. Although existing defense methods have shown some effectiveness, they often struggle to strike a balance between model generality and security. Excessive defense may limit the normal use of the model, while insufficient defense may lead to security vulnerabilities. In response to this problem, we propose a Knowledge Graph Defense Framework (KG-DF). Specifically, because of its structured knowledge representation and semantic association capabilities, Knowledge Graph(KG) can be searched by associating input content with safe knowledge in the knowledge base, thus identifying potentially harmful intentions and providing safe reasoning paths. However, traditional KG methods encounter significant challenges in keyword extraction, particularly when confronted with diverse and evolving attack strategies. To address this issue, we introduce an extensible semantic parsing module, whose core task is to transform the input query into a set of structured and secure concept representations, thereby enhancing the relevance of the matching process. Experimental results show that our framework enhances defense performance against various jailbreak attack methods, while also improving the response quality of the LLM in general QA scenarios by incorporating domain-general knowledge.



## **36. UDora: A Unified Red Teaming Framework against LLM Agents by Dynamically Hijacking Their Own Reasoning**

cs.CR

**SubmitDate**: 2025-11-13    [abs](http://arxiv.org/abs/2503.01908v3) [paper-pdf](None)

**Authors**: Jiawei Zhang, Shuang Yang, Bo Li

**Abstract**: Large Language Model (LLM) agents equipped with external tools have become increasingly powerful for complex tasks such as web shopping, automated email replies, and financial trading. However, these advancements amplify the risks of adversarial attacks, especially when agents can access sensitive external functionalities. Nevertheless, manipulating LLM agents into performing targeted malicious actions or invoking specific tools remains challenging, as these agents extensively reason or plan before executing final actions. In this work, we present UDora, a unified red teaming framework designed for LLM agents that dynamically hijacks the agent's reasoning processes to compel malicious behavior. Specifically, UDora first generates the model's reasoning trace for the given task, then automatically identifies optimal points within this trace to insert targeted perturbations. The resulting perturbed reasoning is then used as a surrogate response for optimization. By iteratively applying this process, the LLM agent will then be induced to undertake designated malicious actions or to invoke specific malicious tools. Our approach demonstrates superior effectiveness compared to existing methods across three LLM agent datasets. The code is available at https://github.com/AI-secure/UDora.



## **37. MPMA: Preference Manipulation Attack Against Model Context Protocol**

cs.CR

This is an extended version of the copyrighted publication at AAAI

**SubmitDate**: 2025-11-12    [abs](http://arxiv.org/abs/2505.11154v2) [paper-pdf](None)

**Authors**: Zihan Wang, Rui Zhang, Yu Liu, Wenshu Fan, Wenbo Jiang, Qingchuan Zhao, Hongwei Li, Guowen Xu

**Abstract**: Model Context Protocol (MCP) standardizes interface mapping for large language models (LLMs) to access external data and tools, which revolutionizes the paradigm of tool selection and facilitates the rapid expansion of the LLM agent tool ecosystem. However, as the MCP is increasingly adopted, third-party customized versions of the MCP server expose potential security vulnerabilities. In this paper, we first introduce a novel security threat, which we term the MCP Preference Manipulation Attack (MPMA). An attacker deploys a customized MCP server to manipulate LLMs, causing them to prioritize it over other competing MCP servers. This can result in economic benefits for attackers, such as revenue from paid MCP services or advertising income generated from free servers. To achieve MPMA, we first design a Direct Preference Manipulation Attack (DPMA) that achieves significant effectiveness by inserting the manipulative word and phrases into the tool name and description. However, such a direct modification is obvious to users and lacks stealthiness. To address these limitations, we further propose Genetic-based Advertising Preference Manipulation Attack (GAPMA). GAPMA employs four commonly used strategies to initialize descriptions and integrates a Genetic Algorithm (GA) to enhance stealthiness. The experiment results demonstrate that GAPMA balances high effectiveness and stealthiness. Our study reveals a critical vulnerability of the MCP in open ecosystems, highlighting an urgent need for robust defense mechanisms to ensure the fairness of the MCP ecosystem.



## **38. Hail to the Thief: Exploring Attacks and Defenses in Decentralised GRPO**

cs.LG

**SubmitDate**: **NEW** 2025-11-14    [abs](http://arxiv.org/abs/2511.09780v1) [paper-pdf](None)

**Authors**: Nikolay Blagoev, Oğuzhan Ersoy, Lydia Yiyu Chen

**Abstract**: Group Relative Policy Optimization (GRPO) has demonstrated great utilization in post-training of Large Language Models (LLMs). In GRPO, prompts are answered by the model and, through reinforcement learning, preferred completions are learnt. Owing to the small communication volume, GRPO is inherently suitable for decentralised training as the prompts can be concurrently answered by multiple nodes and then exchanged in the forms of strings. In this work, we present the first adversarial attack in decentralised GRPO. We demonstrate that malicious parties can poison such systems by injecting arbitrary malicious tokens in benign models in both out-of-context and in-context attacks. Using empirical examples of math and coding tasks, we show that adversarial attacks can easily poison the benign nodes, polluting their local LLM post-training, achieving attack success rates up to 100% in as few as 50 iterations. We propose two ways to defend against these attacks, depending on whether all users train the same model or different models. We show that these defenses can achieve stop rates of up to 100%, making the attack impossible.



## **39. Graph of Attacks with Pruning: Optimizing Stealthy Jailbreak Prompt Generation for Enhanced LLM Content Moderation**

cs.CR

14 pages, 5 figures; published in EMNLP 2025 ; Code at: https://github.com/dsbuddy/GAP-LLM-Safety

**SubmitDate**: **NEW** 2025-11-14    [abs](http://arxiv.org/abs/2501.18638v3) [paper-pdf](None)

**Authors**: Daniel Schwartz, Dmitriy Bespalov, Zhe Wang, Ninad Kulkarni, Yanjun Qi

**Abstract**: As large language models (LLMs) become increasingly prevalent, ensuring their robustness against adversarial misuse is crucial. This paper introduces the GAP (Graph of Attacks with Pruning) framework, an advanced approach for generating stealthy jailbreak prompts to evaluate and enhance LLM safeguards. GAP addresses limitations in existing tree-based LLM jailbreak methods by implementing an interconnected graph structure that enables knowledge sharing across attack paths. Our experimental evaluation demonstrates GAP's superiority over existing techniques, achieving a 20.8% increase in attack success rates while reducing query costs by 62.7%. GAP consistently outperforms state-of-the-art methods for attacking both open and closed LLMs, with attack success rates of >96%. Additionally, we present specialized variants like GAP-Auto for automated seed generation and GAP-VLM for multimodal attacks. GAP-generated prompts prove highly effective in improving content moderation systems, increasing true positive detection rates by 108.5% and accuracy by 183.6% when used for fine-tuning. Our implementation is available at https://github.com/dsbuddy/GAP-LLM-Safety.



## **40. Efficient LLM Safety Evaluation through Multi-Agent Debate**

cs.AI

9 pages of main text, 14 pages total, 4 figures

**SubmitDate**: 2025-11-11    [abs](http://arxiv.org/abs/2511.06396v1) [paper-pdf](None)

**Authors**: Dachuan Lin, Guobin Shen, Zihao Yang, Tianrong Liu, Dongcheng Zhao, Yi Zeng

**Abstract**: Safety evaluation of large language models (LLMs) increasingly relies on LLM-as-a-Judge frameworks, but the high cost of frontier models limits scalability. We propose a cost-efficient multi-agent judging framework that employs Small Language Models (SLMs) through structured debates among critic, defender, and judge agents. To rigorously assess safety judgments, we construct HAJailBench, a large-scale human-annotated jailbreak benchmark comprising 12,000 adversarial interactions across diverse attack methods and target models. The dataset provides fine-grained, expert-labeled ground truth for evaluating both safety robustness and judge reliability. Our SLM-based framework achieves agreement comparable to GPT-4o judges on HAJailBench while substantially reducing inference cost. Ablation results show that three rounds of debate yield the optimal balance between accuracy and efficiency. These findings demonstrate that structured, value-aligned debate enables SLMs to capture semantic nuances of jailbreak attacks and that HAJailBench offers a reliable foundation for scalable LLM safety evaluation.



## **41. Why does weak-OOD help? A Further Step Towards Understanding Jailbreaking VLMs**

cs.CR

**SubmitDate**: 2025-11-12    [abs](http://arxiv.org/abs/2511.08367v1) [paper-pdf](None)

**Authors**: Yuxuan Zhou, Yuzhao Peng, Yang Bai, Kuofeng Gao, Yihao Zhang, Yechao Zhang, Xun Chen, Tao Yu, Tao Dai, Shu-Tao Xia

**Abstract**: Large Vision-Language Models (VLMs) are susceptible to jailbreak attacks: researchers have developed a variety of attack strategies that can successfully bypass the safety mechanisms of VLMs. Among these approaches, jailbreak methods based on the Out-of-Distribution (OOD) strategy have garnered widespread attention due to their simplicity and effectiveness. This paper further advances the in-depth understanding of OOD-based VLM jailbreak methods. Experimental results demonstrate that jailbreak samples generated via mild OOD strategies exhibit superior performance in circumventing the safety constraints of VLMs--a phenomenon we define as ''weak-OOD''. To unravel the underlying causes of this phenomenon, this study takes SI-Attack, a typical OOD-based jailbreak method, as the research object. We attribute this phenomenon to a trade-off between two dominant factors: input intent perception and model refusal triggering. The inconsistency in how these two factors respond to OOD manipulations gives rise to this phenomenon. Furthermore, we provide a theoretical argument for the inevitability of such inconsistency from the perspective of discrepancies between model pre-training and alignment processes. Building on the above insights, we draw inspiration from optical character recognition (OCR) capability enhancement--a core task in the pre-training phase of mainstream VLMs. Leveraging this capability, we design a simple yet highly effective VLM jailbreak method, whose performance outperforms that of SOTA baselines.



## **42. Differentiated Directional Intervention A Framework for Evading LLM Safety Alignment**

cs.CR

AAAI-26-AIA

**SubmitDate**: 2025-11-12    [abs](http://arxiv.org/abs/2511.06852v2) [paper-pdf](None)

**Authors**: Peng Zhang, Peijie Sun

**Abstract**: Safety alignment instills in Large Language Models (LLMs) a critical capacity to refuse malicious requests. Prior works have modeled this refusal mechanism as a single linear direction in the activation space. We posit that this is an oversimplification that conflates two functionally distinct neural processes: the detection of harm and the execution of a refusal. In this work, we deconstruct this single representation into a Harm Detection Direction and a Refusal Execution Direction. Leveraging this fine-grained model, we introduce Differentiated Bi-Directional Intervention (DBDI), a new white-box framework that precisely neutralizes the safety alignment at critical layer. DBDI applies adaptive projection nullification to the refusal execution direction while suppressing the harm detection direction via direct steering. Extensive experiments demonstrate that DBDI outperforms prominent jailbreaking methods, achieving up to a 97.88\% attack success rate on models such as Llama-2. By providing a more granular and mechanistic framework, our work offers a new direction for the in-depth understanding of LLM safety alignment.



## **43. iSeal: Encrypted Fingerprinting for Reliable LLM Ownership Verification**

cs.CR

Accepted by AAAI 2026

**SubmitDate**: 2025-11-13    [abs](http://arxiv.org/abs/2511.08905v1) [paper-pdf](None)

**Authors**: Zixun Xiong, Gaoyi Wu, Qingyang Yu, Mingyu Derek Ma, Lingfeng Yao, Miao Pan, Xiaojiang Du, Hao Wang

**Abstract**: Given the high cost of large language model (LLM) training from scratch, safeguarding LLM intellectual property (IP) has become increasingly crucial. As the standard paradigm for IP ownership verification, LLM fingerprinting thus plays a vital role in addressing this challenge. Existing LLM fingerprinting methods verify ownership by extracting or injecting model-specific features. However, they overlook potential attacks during the verification process, leaving them ineffective when the model thief fully controls the LLM's inference process. In such settings, attackers may share prompt-response pairs to enable fingerprint unlearning or manipulate outputs to evade exact-match verification. We propose iSeal, the first fingerprinting method designed for reliable verification when the model thief controls the suspected LLM in an end-to-end manner. It injects unique features into both the model and an external module, reinforced by an error-correction mechanism and a similarity-based verification strategy. These components are resistant to verification-time attacks, including collusion-based fingerprint unlearning and response manipulation, backed by both theoretical analysis and empirical results. iSeal achieves 100 percent Fingerprint Success Rate (FSR) on 12 LLMs against more than 10 attacks, while baselines fail under unlearning and response manipulations.



## **44. ConfGuard: A Simple and Effective Backdoor Detection for Large Language Models**

cs.CR

This is an extended version of the copyrighted publication at AAAI

**SubmitDate**: 2025-11-12    [abs](http://arxiv.org/abs/2508.01365v3) [paper-pdf](None)

**Authors**: Zihan Wang, Rui Zhang, Hongwei Li, Wenshu Fan, Wenbo Jiang, Qingchuan Zhao, Guowen Xu

**Abstract**: Backdoor attacks pose a significant threat to Large Language Models (LLMs), where adversaries can embed hidden triggers to manipulate LLM's outputs. Most existing defense methods, primarily designed for classification tasks, are ineffective against the autoregressive nature and vast output space of LLMs, thereby suffering from poor performance and high latency. To address these limitations, we investigate the behavioral discrepancies between benign and backdoored LLMs in output space. We identify a critical phenomenon which we term sequence lock: a backdoored model generates the target sequence with abnormally high and consistent confidence compared to benign generation. Building on this insight, we propose ConfGuard, a lightweight and effective detection method that monitors a sliding window of token confidences to identify sequence lock. Extensive experiments demonstrate ConfGuard achieves a near 100\% true positive rate (TPR) and a negligible false positive rate (FPR) in the vast majority of cases. Crucially, the ConfGuard enables real-time detection almost without additional latency, making it a practical backdoor defense for real-world LLM deployments.



## **45. MCP-RiskCue: Can LLM Infer Risk Information From MCP Server System Logs?**

cs.CR

**SubmitDate**: 2025-11-13    [abs](http://arxiv.org/abs/2511.05867v2) [paper-pdf](None)

**Authors**: Jiayi Fu, Qiyao Sun

**Abstract**: Large language models (LLMs) demonstrate strong capabilities in solving complex tasks when integrated with external tools. The Model Context Protocol (MCP) has become a standard interface for enabling such tool-based interactions. However, these interactions introduce substantial security concerns, particularly when the MCP server is compromised or untrustworthy. While prior benchmarks primarily focus on prompt injection attacks or analyze the vulnerabilities of LLM MCP interaction trajectories, limited attention has been given to the underlying system logs associated with malicious MCP servers. To address this gap, we present the first synthetic benchmark for evaluating LLMs ability to identify security risks from system logs. We define nine categories of MCP server risks and generate 1,800 synthetic system logs using ten state-of-the-art LLMs. These logs are embedded in the return values of 243 curated MCP servers, yielding a dataset of 2,421 chat histories for training and 471 queries for evaluation. Our pilot experiments reveal that smaller models often fail to detect risky system logs, leading to high false negatives. While models trained with supervised fine-tuning (SFT) tend to over-flag benign logs, resulting in elevated false positives, Reinforcement Learning from Verifiable Reward (RLVR) offers a better precision-recall balance. In particular, after training with Group Relative Policy Optimization (GRPO), Llama3.1-8B-Instruct achieves 83% accuracy, surpassing the best-performing large remote model by 9 percentage points. Fine-grained, per-category analysis further underscores the effectiveness of reinforcement learning in enhancing LLM safety within the MCP framework. Code and data are available at: https://github.com/PorUna-byte/MCP-RiskCue



## **46. Prompt Injection as an Emerging Threat: Evaluating the Resilience of Large Language Models**

cs.CR

10 pages, 6 figures

**SubmitDate**: 2025-11-13    [abs](http://arxiv.org/abs/2511.01634v2) [paper-pdf](None)

**Authors**: Daniyal Ganiuly, Assel Smaiyl

**Abstract**: Large Language Models (LLMs) are increasingly used in intelligent systems that perform reasoning, summarization, and code generation. Their ability to follow natural-language instructions, while powerful, also makes them vulnerable to a new class of attacks known as prompt injection. In these attacks, hidden or malicious instructions are inserted into user inputs or external content, causing the model to ignore its intended task or produce unsafe responses. This study proposes a unified framework for evaluating how resistant Large Language Models (LLMs) are to prompt injection attacks. The framework defines three complementary metrics such as the Resilience Degradation Index (RDI), Safety Compliance Coefficient (SCC), and Instructional Integrity Metric (IIM) to jointly measure robustness, safety, and semantic stability. We evaluated four instruction-tuned models (GPT-4, GPT-4o, LLaMA-3 8B Instruct, and Flan-T5-Large) on five common language tasks: question answering, summarization, translation, reasoning, and code generation. Results show that GPT-4 performs best overall, while open-weight models remain more vulnerable. The findings highlight that strong alignment and safety tuning are more important for resilience than model size alone. Results show that all models remain partially vulnerable, especially to indirect and direct-override attacks. GPT-4 achieved the best overall resilience (RDR = 9.8 %, SCR = 96.4 %), while open-source models exhibited higher performance degradation and lower safety scores. The findings demonstrate that alignment strength and safety tuning play a greater role in resilience than model size alone. The proposed framework offers a structured, reproducible approach for assessing model robustness and provides practical insights for improving LLM safety and reliability.



## **47. Unlearning Imperative: Securing Trustworthy and Responsible LLMs through Engineered Forgetting**

cs.LG

14 pages, 4 figures, 4 tables

**SubmitDate**: **NEW** 2025-11-14    [abs](http://arxiv.org/abs/2511.09855v1) [paper-pdf](None)

**Authors**: James Jin Kang, Dang Bui, Thanh Pham, Huo-Chong Ling

**Abstract**: The growing use of large language models in sensitive domains has exposed a critical weakness: the inability to ensure that private information can be permanently forgotten. Yet these systems still lack reliable mechanisms to guarantee that sensitive information can be permanently removed once it has been used. Retraining from the beginning is prohibitively costly, and existing unlearning methods remain fragmented, difficult to verify, and often vulnerable to recovery. This paper surveys recent research on machine unlearning for LLMs and considers how far current approaches can address these challenges. We review methods for evaluating whether forgetting has occurred, the resilience of unlearned models against adversarial attacks, and mechanisms that can support user trust when model complexity or proprietary limits restrict transparency. Technical solutions such as differential privacy, homomorphic encryption, federated learning, and ephemeral memory are examined alongside institutional safeguards including auditing practices and regulatory frameworks. The review finds steady progress, but robust and verifiable unlearning is still unresolved. Efficient techniques that avoid costly retraining, stronger defenses against adversarial recovery, and governance structures that reinforce accountability are needed if LLMs are to be deployed safely in sensitive applications. By integrating technical and organizational perspectives, this study outlines a pathway toward AI systems that can be required to forget, while maintaining both privacy and public trust.



## **48. Biologically-Informed Hybrid Membership Inference Attacks on Generative Genomic Models**

cs.CR

**SubmitDate**: **NEW** 2025-11-14    [abs](http://arxiv.org/abs/2511.07503v2) [paper-pdf](None)

**Authors**: Asia Belfiore, Jonathan Passerat-Palmbach, Dmitrii Usynin

**Abstract**: The increased availability of genetic data has transformed genomics research, but raised many privacy concerns regarding its handling due to its sensitive nature. This work explores the use of language models (LMs) for the generation of synthetic genetic mutation profiles, leveraging differential privacy (DP) for the protection of sensitive genetic data. We empirically evaluate the privacy guarantees of our DP modes by introducing a novel Biologically-Informed Hybrid Membership Inference Attack (biHMIA), which combines traditional black box MIA with contextual genomics metrics for enhanced attack power. Our experiments show that both small and large transformer GPT-like models are viable synthetic variant generators for small-scale genomics, and that our hybrid attack leads, on average, to higher adversarial success compared to traditional metric-based MIAs.



## **49. From Capabilities to Performance: Evaluating Key Functional Properties of LLM Architectures in Penetration Testing**

cs.AI

**SubmitDate**: **NEW** 2025-11-14    [abs](http://arxiv.org/abs/2509.14289v3) [paper-pdf](None)

**Authors**: Lanxiao Huang, Daksh Dave, Tyler Cody, Peter Beling, Ming Jin

**Abstract**: Large language models (LLMs) are increasingly used to automate or augment penetration testing, but their effectiveness and reliability across attack phases remain unclear. We present a comprehensive evaluation of multiple LLM-based agents, from single-agent to modular designs, across realistic penetration testing scenarios, measuring empirical performance and recurring failure patterns. We also isolate the impact of five core functional capabilities via targeted augmentations: Global Context Memory (GCM), Inter-Agent Messaging (IAM), Context-Conditioned Invocation (CCI), Adaptive Planning (AP), and Real-Time Monitoring (RTM). These interventions support, respectively: (i) context coherence and retention, (ii) inter-component coordination and state management, (iii) tool use accuracy and selective execution, (iv) multi-step strategic planning, error detection, and recovery, and (v) real-time dynamic responsiveness. Our results show that while some architectures natively exhibit subsets of these properties, targeted augmentations substantially improve modular agent performance, especially in complex, multi-step, and real-time penetration testing tasks.



## **50. Chain-of-Lure: A Universal Jailbreak Attack Framework using Unconstrained Synthetic Narratives**

cs.CR

23 pages, 3 main figures

**SubmitDate**: **NEW** 2025-11-14    [abs](http://arxiv.org/abs/2505.17519v2) [paper-pdf](None)

**Authors**: Wenhan Chang, Tianqing Zhu, Yu Zhao, Shuangyong Song, Ping Xiong, Wanlei Zhou

**Abstract**: In the era of rapid generative AI development, interactions with large language models (LLMs) pose increasing risks of misuse. Prior research has primarily focused on attacks using template-based prompts and optimization-oriented methods, while overlooking the fact that LLMs possess strong unconstrained deceptive capabilities to attack other LLMs. This paper introduces a novel jailbreaking method inspired by the Chain-of-Thought mechanism. The attacker employs mission transfer to conceal harmful user intent within dialogue and generates a progressive chain of lure questions without relying on predefined templates, enabling successful jailbreaks. To further improve the attack's strength, we incorporate a helper LLM model that performs randomized narrative optimization over multi-turn interactions, enhancing the attack performance while preserving alignment with the original intent. We also propose a toxicity-based framework using third-party LLMs to evaluate harmful content and its alignment with malicious intent. Extensive experiments demonstrate that our method consistently achieves high attack success rates and elevated toxicity scores across diverse types of LLMs under black-box API settings. These findings reveal the intrinsic potential of LLMs to perform unrestricted attacks in the absence of robust alignment constraints. Our approach offers data-driven insights to inform the design of future alignment mechanisms. Finally, we propose two concrete defense strategies to support the development of safer generative models.



