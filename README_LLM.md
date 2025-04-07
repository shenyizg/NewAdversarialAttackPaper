# Latest Large Language Model Attack Papers
**update at 2025-04-07 09:22:00**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. sudo rm -rf agentic_security**

cs.CL

**SubmitDate**: 2025-04-04    [abs](http://arxiv.org/abs/2503.20279v2) [paper-pdf](http://arxiv.org/pdf/2503.20279v2)

**Authors**: Sejin Lee, Jian Kim, Haon Park, Ashkan Yousefpour, Sangyoon Yu, Min Song

**Abstract**: Large Language Models (LLMs) are increasingly deployed as computer-use agents, autonomously performing tasks within real desktop or web environments. While this evolution greatly expands practical use cases for humans, it also creates serious security exposures. We present SUDO (Screen-based Universal Detox2Tox Offense), a novel attack framework that systematically bypasses refusal trained safeguards in commercial computer-use agents, such as Claude Computer Use. The core mechanism, Detox2Tox, transforms harmful requests (that agents initially reject) into seemingly benign requests via detoxification, secures detailed instructions from advanced vision language models (VLMs), and then reintroduces malicious content via toxification just before execution. Unlike conventional jailbreaks, SUDO iteratively refines its attacks based on a built-in refusal feedback, making it increasingly effective against robust policy filters. In extensive tests spanning 50 real-world tasks and multiple state-of-the-art VLMs, SUDO achieves a stark attack success rate of 24% (with no refinement), and up to 41% (by its iterative refinement) in Claude Computer Use. By revealing these vulnerabilities and demonstrating the ease with which they can be exploited in real-world computing environments, this paper highlights an immediate need for robust, context-aware safeguards. WARNING: This paper includes harmful or offensive model outputs Our code is available at: https://github.com/AIM-Intelligence/SUDO.git



## **2. Les Dissonances: Cross-Tool Harvesting and Polluting in Multi-Tool Empowered LLM Agents**

cs.CR

**SubmitDate**: 2025-04-04    [abs](http://arxiv.org/abs/2504.03111v1) [paper-pdf](http://arxiv.org/pdf/2504.03111v1)

**Authors**: Zichuan Li, Jian Cui, Xiaojing Liao, Luyi Xing

**Abstract**: Large Language Model (LLM) agents are autonomous systems powered by LLMs, capable of reasoning and planning to solve problems by leveraging a set of tools. However, the integration of multi-tool capabilities in LLM agents introduces challenges in securely managing tools, ensuring their compatibility, handling dependency relationships, and protecting control flows within LLM agent workflows. In this paper, we present the first systematic security analysis of task control flows in multi-tool-enabled LLM agents. We identify a novel threat, Cross-Tool Harvesting and Polluting (XTHP), which includes multiple attack vectors to first hijack the normal control flows of agent tasks, and then collect and pollute confidential or private information within LLM agent systems. To understand the impact of this threat, we developed Chord, a dynamic scanning tool designed to automatically detect real-world agent tools susceptible to XTHP attacks. Our evaluation of 73 real-world tools from the repositories of two major LLM agent development frameworks, LangChain and LlamaIndex, revealed a significant security concern: 80% of the tools are vulnerable to hijacking attacks, 78% to XTH attacks, and 41% to XTP attacks, highlighting the prevalence of this threat.



## **3. PROMPTFUZZ: Harnessing Fuzzing Techniques for Robust Testing of Prompt Injection in LLMs**

cs.CR

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2409.14729v2) [paper-pdf](http://arxiv.org/pdf/2409.14729v2)

**Authors**: Jiahao Yu, Yangguang Shao, Hanwen Miao, Junzheng Shi

**Abstract**: Large Language Models (LLMs) have gained widespread use in various applications due to their powerful capability to generate human-like text. However, prompt injection attacks, which involve overwriting a model's original instructions with malicious prompts to manipulate the generated text, have raised significant concerns about the security and reliability of LLMs. Ensuring that LLMs are robust against such attacks is crucial for their deployment in real-world applications, particularly in critical tasks.   In this paper, we propose PROMPTFUZZ, a novel testing framework that leverages fuzzing techniques to systematically assess the robustness of LLMs against prompt injection attacks. Inspired by software fuzzing, PROMPTFUZZ selects promising seed prompts and generates a diverse set of prompt injections to evaluate the target LLM's resilience. PROMPTFUZZ operates in two stages: the prepare phase, which involves selecting promising initial seeds and collecting few-shot examples, and the focus phase, which uses the collected examples to generate diverse, high-quality prompt injections. Using PROMPTFUZZ, we can uncover more vulnerabilities in LLMs, even those with strong defense prompts.   By deploying the generated attack prompts from PROMPTFUZZ in a real-world competition, we achieved the 7th ranking out of over 4000 participants (top 0.14%) within 2 hours. Additionally, we construct a dataset to fine-tune LLMs for enhanced robustness against prompt injection attacks. While the fine-tuned model shows improved robustness, PROMPTFUZZ continues to identify vulnerabilities, highlighting the importance of robust testing for LLMs. Our work emphasizes the critical need for effective testing tools and provides a practical framework for evaluating and improving the robustness of LLMs against prompt injection attacks.



## **4. ERPO: Advancing Safety Alignment via Ex-Ante Reasoning Preference Optimization**

cs.CL

18 pages, 5 figures

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2504.02725v1) [paper-pdf](http://arxiv.org/pdf/2504.02725v1)

**Authors**: Kehua Feng, Keyan Ding, Jing Yu, Menghan Li, Yuhao Wang, Tong Xu, Xinda Wang, Qiang Zhang, Huajun Chen

**Abstract**: Recent advancements in large language models (LLMs) have accelerated progress toward artificial general intelligence, yet their potential to generate harmful content poses critical safety challenges. Existing alignment methods often struggle to cover diverse safety scenarios and remain vulnerable to adversarial attacks. In this work, we propose Ex-Ante Reasoning Preference Optimization (ERPO), a novel safety alignment framework that equips LLMs with explicit preemptive reasoning through Chain-of-Thought and provides clear evidence for safety judgments by embedding predefined safety rules. Specifically, our approach consists of three stages: first, equipping the model with Ex-Ante reasoning through supervised fine-tuning (SFT) using a constructed reasoning module; second, enhancing safety, usefulness, and efficiency via Direct Preference Optimization (DPO); and third, mitigating inference latency with a length-controlled iterative preference optimization strategy. Experiments on multiple open-source LLMs demonstrate that ERPO significantly enhances safety performance while maintaining response efficiency.



## **5. No Free Lunch with Guardrails**

cs.CR

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2504.00441v2) [paper-pdf](http://arxiv.org/pdf/2504.00441v2)

**Authors**: Divyanshu Kumar, Nitin Aravind Birur, Tanay Baswa, Sahil Agarwal, Prashanth Harshangi

**Abstract**: As large language models (LLMs) and generative AI become widely adopted, guardrails have emerged as a key tool to ensure their safe use. However, adding guardrails isn't without tradeoffs; stronger security measures can reduce usability, while more flexible systems may leave gaps for adversarial attacks. In this work, we explore whether current guardrails effectively prevent misuse while maintaining practical utility. We introduce a framework to evaluate these tradeoffs, measuring how different guardrails balance risk, security, and usability, and build an efficient guardrail.   Our findings confirm that there is no free lunch with guardrails; strengthening security often comes at the cost of usability. To address this, we propose a blueprint for designing better guardrails that minimize risk while maintaining usability. We evaluate various industry guardrails, including Azure Content Safety, Bedrock Guardrails, OpenAI's Moderation API, Guardrails AI, Nemo Guardrails, and Enkrypt AI guardrails. Additionally, we assess how LLMs like GPT-4o, Gemini 2.0-Flash, Claude 3.5-Sonnet, and Mistral Large-Latest respond under different system prompts, including simple prompts, detailed prompts, and detailed prompts with chain-of-thought (CoT) reasoning. Our study provides a clear comparison of how different guardrails perform, highlighting the challenges in balancing security and usability.



## **6. Retrieval-Augmented Purifier for Robust LLM-Empowered Recommendation**

cs.IR

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2504.02458v1) [paper-pdf](http://arxiv.org/pdf/2504.02458v1)

**Authors**: Liangbo Ning, Wenqi Fan, Qing Li

**Abstract**: Recently, Large Language Model (LLM)-empowered recommender systems have revolutionized personalized recommendation frameworks and attracted extensive attention. Despite the remarkable success, existing LLM-empowered RecSys have been demonstrated to be highly vulnerable to minor perturbations. To mitigate the negative impact of such vulnerabilities, one potential solution is to employ collaborative signals based on item-item co-occurrence to purify the malicious collaborative knowledge from the user's historical interactions inserted by attackers. On the other hand, due to the capabilities to expand insufficient internal knowledge of LLMs, Retrieval-Augmented Generation (RAG) techniques provide unprecedented opportunities to enhance the robustness of LLM-empowered recommender systems by introducing external collaborative knowledge. Therefore, in this paper, we propose a novel framework (RETURN) by retrieving external collaborative signals to purify the poisoned user profiles and enhance the robustness of LLM-empowered RecSys in a plug-and-play manner. Specifically, retrieval-augmented perturbation positioning is proposed to identify potential perturbations within the users' historical sequences by retrieving external knowledge from collaborative item graphs. After that, we further retrieve the collaborative knowledge to cleanse the perturbations by using either deletion or replacement strategies and introduce a robust ensemble recommendation strategy to generate final robust predictions. Extensive experiments on three real-world datasets demonstrate the effectiveness of the proposed RETURN.



## **7. ToxicSQL: Migrating SQL Injection Threats into Text-to-SQL Models via Backdoor Attack**

cs.CR

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2503.05445v2) [paper-pdf](http://arxiv.org/pdf/2503.05445v2)

**Authors**: Meiyu Lin, Haichuan Zhang, Jiale Lao, Renyuan Li, Yuanchun Zhou, Carl Yang, Yang Cao, Mingjie Tang

**Abstract**: Large language models (LLMs) have shown state-of-the-art results in translating natural language questions into SQL queries (Text-to-SQL), a long-standing challenge within the database community. However, security concerns remain largely unexplored, particularly the threat of backdoor attacks, which can introduce malicious behaviors into models through fine-tuning with poisoned datasets. In this work, we systematically investigate the vulnerabilities of LLM-based Text-to-SQL models and present ToxicSQL, a novel backdoor attack framework. Our approach leverages stealthy {semantic and character-level triggers} to make backdoors difficult to detect and remove, ensuring that malicious behaviors remain covert while maintaining high model accuracy on benign inputs. Furthermore, we propose leveraging SQL injection payloads as backdoor targets, enabling the generation of malicious yet executable SQL queries, which pose severe security and privacy risks in language model-based SQL development. We demonstrate that injecting only 0.44% of poisoned data can result in an attack success rate of 79.41%, posing a significant risk to database security. Additionally, we propose detection and mitigation strategies to enhance model reliability. Our findings highlight the urgent need for security-aware Text-to-SQL development, emphasizing the importance of robust defenses against backdoor threats.



## **8. Evolving from Single-modal to Multi-modal Facial Deepfake Detection: Progress and Challenges**

cs.CV

P. Liu is with the Department of Computer Science and Engineering,  University of Nevada, Reno, NV, 89512. Q. Tao and J. Zhou are with Centre for  Frontier AI Research (CFAR), and Institute of High Performance Computing  (IHPC), A*STAR, Singapore. J. Zhou is also with Centre for Advanced  Technologies in Online Safety (CATOS), A*STAR, Singapore. J. Zhou is the  corresponding author

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2406.06965v4) [paper-pdf](http://arxiv.org/pdf/2406.06965v4)

**Authors**: Ping Liu, Qiqi Tao, Joey Tianyi Zhou

**Abstract**: As synthetic media, including video, audio, and text, become increasingly indistinguishable from real content, the risks of misinformation, identity fraud, and social manipulation escalate. This survey traces the evolution of deepfake detection from early single-modal methods to sophisticated multi-modal approaches that integrate audio-visual and text-visual cues. We present a structured taxonomy of detection techniques and analyze the transition from GAN-based to diffusion model-driven deepfakes, which introduce new challenges due to their heightened realism and robustness against detection. Unlike prior surveys that primarily focus on single-modal detection or earlier deepfake techniques, this work provides the most comprehensive study to date, encompassing the latest advancements in multi-modal deepfake detection, generalization challenges, proactive defense mechanisms, and emerging datasets specifically designed to support new interpretability and reasoning tasks. We further explore the role of Vision-Language Models (VLMs) and Multimodal Large Language Models (MLLMs) in strengthening detection robustness against increasingly sophisticated deepfake attacks. By systematically categorizing existing methods and identifying emerging research directions, this survey serves as a foundation for future advancements in combating AI-generated facial forgeries. A curated list of all related papers can be found at \href{https://github.com/qiqitao77/Comprehensive-Advances-in-Deepfake-Detection-Spanning-Diverse-Modalities}{https://github.com/qiqitao77/Awesome-Comprehensive-Deepfake-Detection}.



## **9. More is Less: The Pitfalls of Multi-Model Synthetic Preference Data in DPO Safety Alignment**

cs.AI

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2504.02193v1) [paper-pdf](http://arxiv.org/pdf/2504.02193v1)

**Authors**: Yifan Wang, Runjin Chen, Bolian Li, David Cho, Yihe Deng, Ruqi Zhang, Tianlong Chen, Zhangyang Wang, Ananth Grama, Junyuan Hong

**Abstract**: Aligning large language models (LLMs) with human values is an increasingly critical step in post-training. Direct Preference Optimization (DPO) has emerged as a simple, yet effective alternative to reinforcement learning from human feedback (RLHF). Synthetic preference data with its low cost and high quality enable effective alignment through single- or multi-model generated preference data. Our study reveals a striking, safety-specific phenomenon associated with DPO alignment: Although multi-model generated data enhances performance on general tasks (ARC, Hellaswag, MMLU, TruthfulQA, Winogrande) by providing diverse responses, it also tends to facilitate reward hacking during training. This can lead to a high attack success rate (ASR) when models encounter jailbreaking prompts. The issue is particularly pronounced when employing stronger models like GPT-4o or larger models in the same family to generate chosen responses paired with target model self-generated rejected responses, resulting in dramatically poorer safety outcomes. Furthermore, with respect to safety, using solely self-generated responses (single-model generation) for both chosen and rejected pairs significantly outperforms configurations that incorporate responses from stronger models, whether used directly as chosen data or as part of a multi-model response pool. We demonstrate that multi-model preference data exhibits high linear separability between chosen and rejected responses, which allows models to exploit superficial cues rather than internalizing robust safety constraints. Our experiments, conducted on models from the Llama, Mistral, and Qwen families, consistently validate these findings.



## **10. Defending Large Language Models Against Attacks With Residual Stream Activation Analysis**

cs.CR

Included in Proceedings of the Conference on Applied Machine Learning  in Information Security (CAMLIS 2024), Arlington, Virginia, USA, October  24-25, 2024

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2406.03230v5) [paper-pdf](http://arxiv.org/pdf/2406.03230v5)

**Authors**: Amelia Kawasaki, Andrew Davis, Houssam Abbas

**Abstract**: The widespread adoption of Large Language Models (LLMs), exemplified by OpenAI's ChatGPT, brings to the forefront the imperative to defend against adversarial threats on these models. These attacks, which manipulate an LLM's output by introducing malicious inputs, undermine the model's integrity and the trust users place in its outputs. In response to this challenge, our paper presents an innovative defensive strategy, given white box access to an LLM, that harnesses residual activation analysis between transformer layers of the LLM. We apply a novel methodology for analyzing distinctive activation patterns in the residual streams for attack prompt classification. We curate multiple datasets to demonstrate how this method of classification has high accuracy across multiple types of attack scenarios, including our newly-created attack dataset. Furthermore, we enhance the model's resilience by integrating safety fine-tuning techniques for LLMs in order to measure its effect on our capability to detect attacks. The results underscore the effectiveness of our approach in enhancing the detection and mitigation of adversarial inputs, advancing the security framework within which LLMs operate.



## **11. One Pic is All it Takes: Poisoning Visual Document Retrieval Augmented Generation with a Single Image**

cs.CL

8 pages, 6 figures

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2504.02132v1) [paper-pdf](http://arxiv.org/pdf/2504.02132v1)

**Authors**: Ezzeldin Shereen, Dan Ristea, Burak Hasircioglu, Shae McFadden, Vasilios Mavroudis, Chris Hicks

**Abstract**: Multimodal retrieval augmented generation (M-RAG) has recently emerged as a method to inhibit hallucinations of large multimodal models (LMMs) through a factual knowledge base (KB). However, M-RAG also introduces new attack vectors for adversaries that aim to disrupt the system by injecting malicious entries into the KB. In this work, we present a poisoning attack against M-RAG targeting visual document retrieval applications, where the KB contains images of document pages. Our objective is to craft a single image that is retrieved for a variety of different user queries, and consistently influences the output produced by the generative model, thus creating a universal denial-of-service (DoS) attack against the M-RAG system. We demonstrate that while our attack is effective against a diverse range of widely-used, state-of-the-art retrievers (embedding models) and generators (LMMs), it can also be ineffective against robust embedding models. Our attack not only highlights the vulnerability of M-RAG pipelines to poisoning attacks, but also sheds light on a fundamental weakness that potentially hinders their performance even in benign settings.



## **12. Evolving Security in LLMs: A Study of Jailbreak Attacks and Defenses**

cs.CR

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2504.02080v1) [paper-pdf](http://arxiv.org/pdf/2504.02080v1)

**Authors**: Zhengchun Shang, Wenlan Wei

**Abstract**: Large Language Models (LLMs) are increasingly popular, powering a wide range of applications. Their widespread use has sparked concerns, especially through jailbreak attacks that bypass safety measures to produce harmful content.   In this paper, we present a comprehensive security analysis of large language models (LLMs), addressing critical research questions on the evolution and determinants of model safety.   Specifically, we begin by identifying the most effective techniques for detecting jailbreak attacks. Next, we investigate whether newer versions of LLMs offer improved security compared to their predecessors. We also assess the impact of model size on overall security and explore the potential benefits of integrating multiple defense strategies to enhance model robustness.   Our study evaluates both open-source models (e.g., LLaMA and Mistral) and closed-source systems (e.g., GPT-4) by employing four state-of-the-art attack techniques and assessing the efficacy of three new defensive approaches.



## **13. AdPO: Enhancing the Adversarial Robustness of Large Vision-Language Models with Preference Optimization**

cs.CV

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2504.01735v1) [paper-pdf](http://arxiv.org/pdf/2504.01735v1)

**Authors**: Chaohu Liu, Tianyi Gui, Yu Liu, Linli Xu

**Abstract**: Large Vision-Language Models (LVLMs), such as GPT-4o and LLaVA, have recently witnessed remarkable advancements and are increasingly being deployed in real-world applications. However, inheriting the sensitivity of visual neural networks, LVLMs remain vulnerable to adversarial attacks, which can result in erroneous or malicious outputs. While existing efforts utilize adversarial fine-tuning to enhance robustness, they often suffer from performance degradation on clean inputs. In this paper, we proposes AdPO, a novel adversarial defense strategy for LVLMs based on preference optimization. For the first time, we reframe adversarial training as a preference optimization problem, aiming to enhance the model's preference for generating normal outputs on clean inputs while rejecting the potential misleading outputs for adversarial examples. Notably, AdPO achieves this by solely modifying the image encoder, e.g., CLIP ViT, resulting in superior clean and adversarial performance in a variety of downsream tasks. Considering that training involves large language models (LLMs), the computational cost increases significantly. We validate that training on smaller LVLMs and subsequently transferring to larger models can achieve competitive performance while maintaining efficiency comparable to baseline methods. Our comprehensive experiments confirm the effectiveness of the proposed AdPO, which provides a novel perspective for future adversarial defense research.



## **14. Representation Bending for Large Language Model Safety**

cs.LG

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2504.01550v1) [paper-pdf](http://arxiv.org/pdf/2504.01550v1)

**Authors**: Ashkan Yousefpour, Taeheon Kim, Ryan S. Kwon, Seungbeen Lee, Wonje Jeung, Seungju Han, Alvin Wan, Harrison Ngan, Youngjae Yu, Jonghyun Choi

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools, but their inherent safety risks - ranging from harmful content generation to broader societal harms - pose significant challenges. These risks can be amplified by the recent adversarial attacks, fine-tuning vulnerabilities, and the increasing deployment of LLMs in high-stakes environments. Existing safety-enhancing techniques, such as fine-tuning with human feedback or adversarial training, are still vulnerable as they address specific threats and often fail to generalize across unseen attacks, or require manual system-level defenses. This paper introduces RepBend, a novel approach that fundamentally disrupts the representations underlying harmful behaviors in LLMs, offering a scalable solution to enhance (potentially inherent) safety. RepBend brings the idea of activation steering - simple vector arithmetic for steering model's behavior during inference - to loss-based fine-tuning. Through extensive evaluation, RepBend achieves state-of-the-art performance, outperforming prior methods such as Circuit Breaker, RMU, and NPO, with up to 95% reduction in attack success rates across diverse jailbreak benchmarks, all with negligible reduction in model usability and general capabilities.



## **15. LightDefense: A Lightweight Uncertainty-Driven Defense against Jailbreaks via Shifted Token Distribution**

cs.CR

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2504.01533v1) [paper-pdf](http://arxiv.org/pdf/2504.01533v1)

**Authors**: Zhuoran Yang, Jie Peng, Zhen Tan, Tianlong Chen, Yanyong Zhang

**Abstract**: Large Language Models (LLMs) face threats from jailbreak prompts. Existing methods for defending against jailbreak attacks are primarily based on auxiliary models. These strategies, however, often require extensive data collection or training. We propose LightDefense, a lightweight defense mechanism targeted at white-box models, which utilizes a safety-oriented direction to adjust the probabilities of tokens in the vocabulary, making safety disclaimers appear among the top tokens after sorting tokens by probability in descending order. We further innovatively leverage LLM's uncertainty about prompts to measure their harmfulness and adaptively adjust defense strength, effectively balancing safety and helpfulness. The effectiveness of LightDefense in defending against 5 attack methods across 2 target LLMs, without compromising helpfulness to benign user queries, highlights its potential as a novel and lightweight defense mechanism, enhancing security of LLMs.



## **16. PiCo: Jailbreaking Multimodal Large Language Models via $\textbf{Pi}$ctorial $\textbf{Co}$de Contextualization**

cs.CR

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2504.01444v1) [paper-pdf](http://arxiv.org/pdf/2504.01444v1)

**Authors**: Aofan Liu, Lulu Tang, Ting Pan, Yuguo Yin, Bin Wang, Ao Yang

**Abstract**: Multimodal Large Language Models (MLLMs), which integrate vision and other modalities into Large Language Models (LLMs), significantly enhance AI capabilities but also introduce new security vulnerabilities. By exploiting the vulnerabilities of the visual modality and the long-tail distribution characteristic of code training data, we present PiCo, a novel jailbreaking framework designed to progressively bypass multi-tiered defense mechanisms in advanced MLLMs. PiCo employs a tier-by-tier jailbreak strategy, using token-level typographic attacks to evade input filtering and embedding harmful intent within programming context instructions to bypass runtime monitoring. To comprehensively assess the impact of attacks, a new evaluation metric is further proposed to assess both the toxicity and helpfulness of model outputs post-attack. By embedding harmful intent within code-style visual instructions, PiCo achieves an average Attack Success Rate (ASR) of 84.13% on Gemini-Pro Vision and 52.66% on GPT-4, surpassing previous methods. Experimental results highlight the critical gaps in current defenses, underscoring the need for more robust strategies to secure advanced MLLMs.



## **17. Safeguarding Vision-Language Models: Mitigating Vulnerabilities to Gaussian Noise in Perturbation-based Attacks**

cs.CV

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2504.01308v1) [paper-pdf](http://arxiv.org/pdf/2504.01308v1)

**Authors**: Jiawei Wang, Yushen Zuo, Yuanjun Chai, Zhendong Liu, Yichen Fu, Yichun Feng, Kin-man Lam

**Abstract**: Vision-Language Models (VLMs) extend the capabilities of Large Language Models (LLMs) by incorporating visual information, yet they remain vulnerable to jailbreak attacks, especially when processing noisy or corrupted images. Although existing VLMs adopt security measures during training to mitigate such attacks, vulnerabilities associated with noise-augmented visual inputs are overlooked. In this work, we identify that missing noise-augmented training causes critical security gaps: many VLMs are susceptible to even simple perturbations such as Gaussian noise. To address this challenge, we propose Robust-VLGuard, a multimodal safety dataset with aligned / misaligned image-text pairs, combined with noise-augmented fine-tuning that reduces attack success rates while preserving functionality of VLM. For stronger optimization-based visual perturbation attacks, we propose DiffPure-VLM, leveraging diffusion models to convert adversarial perturbations into Gaussian-like noise, which can be defended by VLMs with noise-augmented safety fine-tuning. Experimental results demonstrate that the distribution-shifting property of diffusion model aligns well with our fine-tuned VLMs, significantly mitigating adversarial perturbations across varying intensities. The dataset and code are available at https://github.com/JarvisUSTC/DiffPure-RobustVLM.



## **18. Strategize Globally, Adapt Locally: A Multi-Turn Red Teaming Agent with Dual-Level Learning**

cs.AI

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2504.01278v1) [paper-pdf](http://arxiv.org/pdf/2504.01278v1)

**Authors**: Si Chen, Xiao Yu, Ninareh Mehrabi, Rahul Gupta, Zhou Yu, Ruoxi Jia

**Abstract**: The exploitation of large language models (LLMs) for malicious purposes poses significant security risks as these models become more powerful and widespread. While most existing red-teaming frameworks focus on single-turn attacks, real-world adversaries typically operate in multi-turn scenarios, iteratively probing for vulnerabilities and adapting their prompts based on threat model responses. In this paper, we propose \AlgName, a novel multi-turn red-teaming agent that emulates sophisticated human attackers through complementary learning dimensions: global tactic-wise learning that accumulates knowledge over time and generalizes to new attack goals, and local prompt-wise learning that refines implementations for specific goals when initial attempts fail. Unlike previous multi-turn approaches that rely on fixed strategy sets, \AlgName enables the agent to identify new jailbreak tactics, develop a goal-based tactic selection framework, and refine prompt formulations for selected tactics. Empirical evaluations on JailbreakBench demonstrate our framework's superior performance, achieving over 90\% attack success rates against GPT-3.5-Turbo and Llama-3.1-70B within 5 conversation turns, outperforming state-of-the-art baselines. These results highlight the effectiveness of dynamic learning in identifying and exploiting model vulnerabilities in realistic multi-turn scenarios.



## **19. Towards Resilient Federated Learning in CyberEdge Networks: Recent Advances and Future Trends**

cs.CR

15 pages, 8 figures, 4 tables, 122 references, journal paper

**SubmitDate**: 2025-04-01    [abs](http://arxiv.org/abs/2504.01240v1) [paper-pdf](http://arxiv.org/pdf/2504.01240v1)

**Authors**: Kai Li, Zhengyang Zhang, Azadeh Pourkabirian, Wei Ni, Falko Dressler, Ozgur B. Akan

**Abstract**: In this survey, we investigate the most recent techniques of resilient federated learning (ResFL) in CyberEdge networks, focusing on joint training with agglomerative deduction and feature-oriented security mechanisms. We explore adaptive hierarchical learning strategies to tackle non-IID data challenges, improving scalability and reducing communication overhead. Fault tolerance techniques and agglomerative deduction mechanisms are studied to detect unreliable devices, refine model updates, and enhance convergence stability. Unlike existing FL security research, we comprehensively analyze feature-oriented threats, such as poisoning, inference, and reconstruction attacks that exploit model features. Moreover, we examine resilient aggregation techniques, anomaly detection, and cryptographic defenses, including differential privacy and secure multi-party computation, to strengthen FL security. In addition, we discuss the integration of 6G, large language models (LLMs), and interoperable learning frameworks to enhance privacy-preserving and decentralized cross-domain training. These advancements offer ultra-low latency, artificial intelligence (AI)-driven network management, and improved resilience against adversarial attacks, fostering the deployment of secure ResFL in CyberEdge networks.



## **20. Multilingual and Multi-Accent Jailbreaking of Audio LLMs**

cs.SD

21 pages, 6 figures, 15 tables

**SubmitDate**: 2025-04-01    [abs](http://arxiv.org/abs/2504.01094v1) [paper-pdf](http://arxiv.org/pdf/2504.01094v1)

**Authors**: Jaechul Roh, Virat Shejwalkar, Amir Houmansadr

**Abstract**: Large Audio Language Models (LALMs) have significantly advanced audio understanding but introduce critical security risks, particularly through audio jailbreaks. While prior work has focused on English-centric attacks, we expose a far more severe vulnerability: adversarial multilingual and multi-accent audio jailbreaks, where linguistic and acoustic variations dramatically amplify attack success. In this paper, we introduce Multi-AudioJail, the first systematic framework to exploit these vulnerabilities through (1) a novel dataset of adversarially perturbed multilingual/multi-accent audio jailbreaking prompts, and (2) a hierarchical evaluation pipeline revealing that how acoustic perturbations (e.g., reverberation, echo, and whisper effects) interacts with cross-lingual phonetics to cause jailbreak success rates (JSRs) to surge by up to +57.25 percentage points (e.g., reverberated Kenyan-accented attack on MERaLiON). Crucially, our work further reveals that multimodal LLMs are inherently more vulnerable than unimodal systems: attackers need only exploit the weakest link (e.g., non-English audio inputs) to compromise the entire model, which we empirically show by multilingual audio-only attacks achieving 3.1x higher success rates than text-only attacks. We plan to release our dataset to spur research into cross-modal defenses, urging the community to address this expanding attack surface in multimodality as LALMs evolve.



## **21. The Illusionist's Prompt: Exposing the Factual Vulnerabilities of Large Language Models with Linguistic Nuances**

cs.CL

work in progress

**SubmitDate**: 2025-04-01    [abs](http://arxiv.org/abs/2504.02865v1) [paper-pdf](http://arxiv.org/pdf/2504.02865v1)

**Authors**: Yining Wang, Yuquan Wang, Xi Li, Mi Zhang, Geng Hong, Min Yang

**Abstract**: As Large Language Models (LLMs) continue to advance, they are increasingly relied upon as real-time sources of information by non-expert users. To ensure the factuality of the information they provide, much research has focused on mitigating hallucinations in LLM responses, but only in the context of formal user queries, rather than maliciously crafted ones. In this study, we introduce The Illusionist's Prompt, a novel hallucination attack that incorporates linguistic nuances into adversarial queries, challenging the factual accuracy of LLMs against five types of fact-enhancing strategies. Our attack automatically generates highly transferrable illusory prompts to induce internal factual errors, all while preserving user intent and semantics. Extensive experiments confirm the effectiveness of our attack in compromising black-box LLMs, including commercial APIs like GPT-4o and Gemini-2.0, even with various defensive mechanisms.



## **22. Exposing the Ghost in the Transformer: Abnormal Detection for Large Language Models via Hidden State Forensics**

cs.CR

**SubmitDate**: 2025-04-01    [abs](http://arxiv.org/abs/2504.00446v1) [paper-pdf](http://arxiv.org/pdf/2504.00446v1)

**Authors**: Shide Zhou, Kailong Wang, Ling Shi, Haoyu Wang

**Abstract**: The widespread adoption of Large Language Models (LLMs) in critical applications has introduced severe reliability and security risks, as LLMs remain vulnerable to notorious threats such as hallucinations, jailbreak attacks, and backdoor exploits. These vulnerabilities have been weaponized by malicious actors, leading to unauthorized access, widespread misinformation, and compromised LLM-embedded system integrity. In this work, we introduce a novel approach to detecting abnormal behaviors in LLMs via hidden state forensics. By systematically inspecting layer-specific activation patterns, we develop a unified framework that can efficiently identify a range of security threats in real-time without imposing prohibitive computational costs. Extensive experiments indicate detection accuracies exceeding 95% and consistently robust performance across multiple models in most scenarios, while preserving the ability to detect novel attacks effectively. Furthermore, the computational overhead remains minimal, with merely fractions of a second. The significance of this work lies in proposing a promising strategy to reinforce the security of LLM-integrated systems, paving the way for safer and more reliable deployment in high-stakes domains. By enabling real-time detection that can also support the mitigation of abnormal behaviors, it represents a meaningful step toward ensuring the trustworthiness of AI systems amid rising security challenges.



## **23. Unleashing the Power of Pre-trained Encoders for Universal Adversarial Attack Detection**

cs.CV

**SubmitDate**: 2025-04-01    [abs](http://arxiv.org/abs/2504.00429v1) [paper-pdf](http://arxiv.org/pdf/2504.00429v1)

**Authors**: Yinghe Zhang, Chi Liu, Shuai Zhou, Sheng Shen, Peng Gui

**Abstract**: Adversarial attacks pose a critical security threat to real-world AI systems by injecting human-imperceptible perturbations into benign samples to induce misclassification in deep learning models. While existing detection methods, such as Bayesian uncertainty estimation and activation pattern analysis, have achieved progress through feature engineering, their reliance on handcrafted feature design and prior knowledge of attack patterns limits generalization capabilities and incurs high engineering costs. To address these limitations, this paper proposes a lightweight adversarial detection framework based on the large-scale pre-trained vision-language model CLIP. Departing from conventional adversarial feature characterization paradigms, we innovatively adopt an anomaly detection perspective. By jointly fine-tuning CLIP's dual visual-text encoders with trainable adapter networks and learnable prompts, we construct a compact representation space tailored for natural images. Notably, our detection architecture achieves substantial improvements in generalization capability across both known and unknown attack patterns compared to traditional methods, while significantly reducing training overhead. This study provides a novel technical pathway for establishing a parameter-efficient and attack-agnostic defense paradigm, markedly enhancing the robustness of vision systems against evolving adversarial threats.



## **24. Understanding the Effectiveness of Coverage Criteria for Large Language Models: A Special Angle from Jailbreak Attacks**

cs.SE

**SubmitDate**: 2025-04-01    [abs](http://arxiv.org/abs/2408.15207v3) [paper-pdf](http://arxiv.org/pdf/2408.15207v3)

**Authors**: Shide Zhou, Tianlin Li, Kailong Wang, Yihao Huang, Ling Shi, Yang Liu, Haoyu Wang

**Abstract**: Large language models (LLMs) have revolutionized artificial intelligence, but their increasing deployment across critical domains has raised concerns about their abnormal behaviors when faced with malicious attacks. Such vulnerability alerts the widespread inadequacy of pre-release testing. In this paper, we conduct a comprehensive empirical study to evaluate the effectiveness of traditional coverage criteria in identifying such inadequacies, exemplified by the significant security concern of jailbreak attacks. Our study begins with a clustering analysis of the hidden states of LLMs, revealing that the embedded characteristics effectively distinguish between different query types. We then systematically evaluate the performance of these criteria across three key dimensions: criterion level, layer level, and token level. Our research uncovers significant differences in neuron coverage when LLMs process normal versus jailbreak queries, aligning with our clustering experiments. Leveraging these findings, we propose three practical applications of coverage criteria in the context of LLM security testing. Specifically, we develop a real-time jailbreak detection mechanism that achieves high accuracy (93.61% on average) in classifying queries as normal or jailbreak. Furthermore, we explore the use of coverage levels to prioritize test cases, improving testing efficiency by focusing on high-risk interactions and removing redundant tests. Lastly, we introduce a coverage-guided approach for generating jailbreak attack examples, enabling systematic refinement of prompts to uncover vulnerabilities. This study improves our understanding of LLM security testing, enhances their safety, and provides a foundation for developing more robust AI applications.



## **25. BounTCHA: A CAPTCHA Utilizing Boundary Identification in Guided Generative AI-extended Videos**

cs.CR

22 pages, 15 figures; references added, typos corrected, new keyword  "guided" added, new experimental data and related results updated; new  keyword "Generative AI" added for clarity

**SubmitDate**: 2025-04-01    [abs](http://arxiv.org/abs/2501.18565v3) [paper-pdf](http://arxiv.org/pdf/2501.18565v3)

**Authors**: Lehao Lin, Ke Wang, Maha Abdallah, Wei Cai

**Abstract**: In recent years, the rapid development of artificial intelligence (AI) especially multi-modal Large Language Models (MLLMs), has enabled it to understand text, images, videos, and other multimedia data, allowing AI systems to execute various tasks based on human-provided prompts. However, AI-powered bots have increasingly been able to bypass most existing CAPTCHA systems, posing significant security threats to web applications. This makes the design of new CAPTCHA mechanisms an urgent priority. We observe that humans are highly sensitive to shifts and abrupt changes in videos, while current AI systems still struggle to comprehend and respond to such situations effectively. Based on this observation, we design and implement BounTCHA, a CAPTCHA mechanism that leverages human perception of boundaries in video transitions and disruptions. By utilizing generative AI's capability to extend original videos with prompts, we introduce unexpected twists and changes to create a pipeline for generating guided short videos for CAPTCHA purposes. We develop a prototype and conduct experiments to collect data on humans' time biases in boundary identification. This data serves as a basis for distinguishing between human users and bots. Additionally, we perform a detailed security analysis of BounTCHA, demonstrating its resilience against various types of attacks. We hope that BounTCHA will act as a robust defense, safeguarding millions of web applications in the AI-driven era.



## **26. Integrated LLM-Based Intrusion Detection with Secure Slicing xApp for Securing O-RAN-Enabled Wireless Network Deployments**

cs.CR

This article has been accepted for publication in the IEEE 2025  International Conference on Communications (ICC2025)

**SubmitDate**: 2025-04-01    [abs](http://arxiv.org/abs/2504.00341v1) [paper-pdf](http://arxiv.org/pdf/2504.00341v1)

**Authors**: Joshua Moore, Aly Sabri Abdalla, Prabesh Khanal, Vuk Marojevic

**Abstract**: The Open Radio Access Network (O-RAN) architecture is reshaping telecommunications by promoting openness, flexibility, and intelligent closed-loop optimization. By decoupling hardware and software and enabling multi-vendor deployments, O-RAN reduces costs, enhances performance, and allows rapid adaptation to new technologies. A key innovation is intelligent network slicing, which partitions networks into isolated slices tailored for specific use cases or quality of service requirements. The RAN Intelligent Controller further optimizes resource allocation, ensuring efficient utilization and improved service quality for user equipment (UEs). However, the modular and dynamic nature of O-RAN expands the threat surface, necessitating advanced security measures to maintain network integrity, confidentiality, and availability. Intrusion detection systems have become essential for identifying and mitigating attacks. This research explores using large language models (LLMs) to generate security recommendations based on the temporal traffic patterns of connected UEs. The paper introduces an LLM-driven intrusion detection framework and demonstrates its efficacy through experimental deployments, comparing non fine-tuned and fine-tuned models for task-specific accuracy.



## **27. $\textit{Agents Under Siege}$: Breaking Pragmatic Multi-Agent LLM Systems with Optimized Prompt Attacks**

cs.MA

**SubmitDate**: 2025-03-31    [abs](http://arxiv.org/abs/2504.00218v1) [paper-pdf](http://arxiv.org/pdf/2504.00218v1)

**Authors**: Rana Muhammad Shahroz Khan, Zhen Tan, Sukwon Yun, Charles Flemming, Tianlong Chen

**Abstract**: Most discussions about Large Language Model (LLM) safety have focused on single-agent settings but multi-agent LLM systems now create novel adversarial risks because their behavior depends on communication between agents and decentralized reasoning. In this work, we innovatively focus on attacking pragmatic systems that have constrains such as limited token bandwidth, latency between message delivery, and defense mechanisms. We design a $\textit{permutation-invariant adversarial attack}$ that optimizes prompt distribution across latency and bandwidth-constraint network topologies to bypass distributed safety mechanisms within the system. Formulating the attack path as a problem of $\textit{maximum-flow minimum-cost}$, coupled with the novel $\textit{Permutation-Invariant Evasion Loss (PIEL)}$, we leverage graph-based optimization to maximize attack success rate while minimizing detection risk. Evaluating across models including $\texttt{Llama}$, $\texttt{Mistral}$, $\texttt{Gemma}$, $\texttt{DeepSeek}$ and other variants on various datasets like $\texttt{JailBreakBench}$ and $\texttt{AdversarialBench}$, our method outperforms conventional attacks by up to $7\times$, exposing critical vulnerabilities in multi-agent systems. Moreover, we demonstrate that existing defenses, including variants of $\texttt{Llama-Guard}$ and $\texttt{PromptGuard}$, fail to prohibit our attack, emphasizing the urgent need for multi-agent specific safety mechanisms.



## **28. Output Constraints as Attack Surface: Exploiting Structured Generation to Bypass LLM Safety Mechanisms**

cs.CR

15 pages, 13 figures, 4 tables Work In Progress

**SubmitDate**: 2025-03-31    [abs](http://arxiv.org/abs/2503.24191v1) [paper-pdf](http://arxiv.org/pdf/2503.24191v1)

**Authors**: Shuoming Zhang, Jiacheng Zhao, Ruiyuan Xu, Xiaobing Feng, Huimin Cui

**Abstract**: Content Warning: This paper may contain unsafe or harmful content generated by LLMs that may be offensive to readers. Large Language Models (LLMs) are extensively used as tooling platforms through structured output APIs to ensure syntax compliance so that robust integration with existing softwares like agent systems, could be achieved. However, the feature enabling functionality of grammar-guided structured output presents significant security vulnerabilities. In this work, we reveal a critical control-plane attack surface orthogonal to traditional data-plane vulnerabilities. We introduce Constrained Decoding Attack (CDA), a novel jailbreak class that weaponizes structured output constraints to bypass safety mechanisms. Unlike prior attacks focused on input prompts, CDA operates by embedding malicious intent in schema-level grammar rules (control-plane) while maintaining benign surface prompts (data-plane). We instantiate this with a proof-of-concept Chain Enum Attack, achieves 96.2% attack success rates across proprietary and open-weight LLMs on five safety benchmarks with a single query, including GPT-4o and Gemini-2.0-flash. Our findings identify a critical security blind spot in current LLM architectures and urge a paradigm shift in LLM safety to address control-plane vulnerabilities, as current mechanisms focused solely on data-plane threats leave critical systems exposed.



## **29. Get the Agents Drunk: Memory Perturbations in Autonomous Agent-based Recommender Systems**

cs.CR

**SubmitDate**: 2025-03-31    [abs](http://arxiv.org/abs/2503.23804v1) [paper-pdf](http://arxiv.org/pdf/2503.23804v1)

**Authors**: Shiyi Yang, Zhibo Hu, Chen Wang, Tong Yu, Xiwei Xu, Liming Zhu, Lina Yao

**Abstract**: Large language model-based agents are increasingly used in recommender systems (Agent4RSs) to achieve personalized behavior modeling. Specifically, Agent4RSs introduces memory mechanisms that enable the agents to autonomously learn and self-evolve from real-world interactions. However, to the best of our knowledge, how robust Agent4RSs are remains unexplored. As such, in this paper, we propose the first work to attack Agent4RSs by perturbing agents' memories, not only to uncover their limitations but also to enhance their security and robustness, ensuring the development of safer and more reliable AI agents.   Given the security and privacy concerns, it is more practical to launch attacks under a black-box setting, where the accurate knowledge of the victim models cannot be easily obtained. Moreover, the practical attacks are often stealthy to maximize the impact. To this end, we propose a novel practical attack framework named DrunkAgent. DrunkAgent consists of a generation module, a strategy module, and a surrogate module. The generation module aims to produce effective and coherent adversarial textual triggers, which can be used to achieve attack objectives such as promoting the target items. The strategy module is designed to `get the target agents drunk' so that their memories cannot be effectively updated during the interaction process. As such, the triggers can play the best role. Both of the modules are optimized on the surrogate module to improve the transferability and imperceptibility of the attacks. By identifying and analyzing the vulnerabilities, our work provides critical insights that pave the way for building safer and more resilient Agent4RSs. Extensive experiments across various real-world datasets demonstrate the effectiveness of DrunkAgent.



## **30. CL-Attack: Textual Backdoor Attacks via Cross-Lingual Triggers**

cs.CR

The paper has been accepted to AAAI 2025

**SubmitDate**: 2025-03-31    [abs](http://arxiv.org/abs/2412.19037v2) [paper-pdf](http://arxiv.org/pdf/2412.19037v2)

**Authors**: Jingyi Zheng, Tianyi Hu, Tianshuo Cong, Xinlei He

**Abstract**: Backdoor attacks significantly compromise the security of large language models by triggering them to output specific and controlled content. Currently, triggers for textual backdoor attacks fall into two categories: fixed-token triggers and sentence-pattern triggers. However, the former are typically easy to identify and filter, while the latter, such as syntax and style, do not apply to all original samples and may lead to semantic shifts. In this paper, inspired by cross-lingual (CL) prompts of LLMs in real-world scenarios, we propose a higher-dimensional trigger method at the paragraph level, namely CL-attack. CL-attack injects the backdoor by using texts with specific structures that incorporate multiple languages, thereby offering greater stealthiness and universality compared to existing backdoor attack techniques. Extensive experiments on different tasks and model architectures demonstrate that CL-attack can achieve nearly 100% attack success rate with a low poisoning rate in both classification and generation tasks. We also empirically show that the CL-attack is more robust against current major defense methods compared to baseline backdoor attacks. Additionally, to mitigate CL-attack, we further develop a new defense called TranslateDefense, which can partially mitigate the impact of CL-attack.



## **31. InjecGuard: Benchmarking and Mitigating Over-defense in Prompt Injection Guardrail Models**

cs.CL

**SubmitDate**: 2025-03-30    [abs](http://arxiv.org/abs/2410.22770v3) [paper-pdf](http://arxiv.org/pdf/2410.22770v3)

**Authors**: Hao Li, Xiaogeng Liu

**Abstract**: Prompt injection attacks pose a critical threat to large language models (LLMs), enabling goal hijacking and data leakage. Prompt guard models, though effective in defense, suffer from over-defense -- falsely flagging benign inputs as malicious due to trigger word bias. To address this issue, we introduce NotInject, an evaluation dataset that systematically measures over-defense across various prompt guard models. NotInject contains 339 benign samples enriched with trigger words common in prompt injection attacks, enabling fine-grained evaluation. Our results show that state-of-the-art models suffer from over-defense issues, with accuracy dropping close to random guessing levels (60%). To mitigate this, we propose InjecGuard, a novel prompt guard model that incorporates a new training strategy, Mitigating Over-defense for Free (MOF), which significantly reduces the bias on trigger words. InjecGuard demonstrates state-of-the-art performance on diverse benchmarks including NotInject, surpassing the existing best model by 30.8%, offering a robust and open-source solution for detecting prompt injection attacks. The code and datasets are released at https://github.com/leolee99/InjecGuard.



## **32. Data Extraction Attacks in Retrieval-Augmented Generation via Backdoors**

cs.CR

**SubmitDate**: 2025-03-30    [abs](http://arxiv.org/abs/2411.01705v2) [paper-pdf](http://arxiv.org/pdf/2411.01705v2)

**Authors**: Yuefeng Peng, Junda Wang, Hong Yu, Amir Houmansadr

**Abstract**: Despite significant advancements, large language models (LLMs) still struggle with providing accurate answers when lacking domain-specific or up-to-date knowledge. Retrieval-Augmented Generation (RAG) addresses this limitation by incorporating external knowledge bases, but it also introduces new attack surfaces. In this paper, we investigate data extraction attacks targeting RAG's knowledge databases. We show that previous prompt injection-based extraction attacks largely rely on the instruction-following capabilities of LLMs. As a result, they fail on models that are less responsive to such malicious prompts -- for example, our experiments show that state-of-the-art attacks achieve near-zero success on Gemma-2B-IT. Moreover, even for models that can follow these instructions, we found fine-tuning may significantly reduce attack performance. To further reveal the vulnerability, we propose to backdoor RAG, where a small portion of poisoned data is injected during the fine-tuning phase to create a backdoor within the LLM. When this compromised LLM is integrated into a RAG system, attackers can exploit specific triggers in prompts to manipulate the LLM to leak documents from the retrieval database. By carefully designing the poisoned data, we achieve both verbatim and paraphrased document extraction. For example, on Gemma-2B-IT, we show that with only 5\% poisoned data, our method achieves an average success rate of 94.1\% for verbatim extraction (ROUGE-L score: 82.1) and 63.6\% for paraphrased extraction (average ROUGE score: 66.4) across four datasets. These results underscore the privacy risks associated with the supply chain when deploying RAG systems.



## **33. Encrypted Prompt: Securing LLM Applications Against Unauthorized Actions**

cs.CR

**SubmitDate**: 2025-03-29    [abs](http://arxiv.org/abs/2503.23250v1) [paper-pdf](http://arxiv.org/pdf/2503.23250v1)

**Authors**: Shih-Han Chan

**Abstract**: Security threats like prompt injection attacks pose significant risks to applications that integrate Large Language Models (LLMs), potentially leading to unauthorized actions such as API misuse. Unlike previous approaches that aim to detect these attacks on a best-effort basis, this paper introduces a novel method that appends an Encrypted Prompt to each user prompt, embedding current permissions. These permissions are verified before executing any actions (such as API calls) generated by the LLM. If the permissions are insufficient, the LLM's actions will not be executed, ensuring safety. This approach guarantees that only actions within the scope of the current permissions from the LLM can proceed. In scenarios where adversarial prompts are introduced to mislead the LLM, this method ensures that any unauthorized actions from LLM wouldn't be executed by verifying permissions in Encrypted Prompt. Thus, threats like prompt injection attacks that trigger LLM to generate harmful actions can be effectively mitigated.



## **34. Training Large Language Models for Advanced Typosquatting Detection**

cs.CR

6 pages, 1 figure

**SubmitDate**: 2025-03-28    [abs](http://arxiv.org/abs/2503.22406v1) [paper-pdf](http://arxiv.org/pdf/2503.22406v1)

**Authors**: Jackson Welch

**Abstract**: Typosquatting is a long-standing cyber threat that exploits human error in typing URLs to deceive users, distribute malware, and conduct phishing attacks. With the proliferation of domain names and new Top-Level Domains (TLDs), typosquatting techniques have grown more sophisticated, posing significant risks to individuals, businesses, and national cybersecurity infrastructure. Traditional detection methods primarily focus on well-known impersonation patterns, leaving gaps in identifying more complex attacks. This study introduces a novel approach leveraging large language models (LLMs) to enhance typosquatting detection. By training an LLM on character-level transformations and pattern-based heuristics rather than domain-specific data, a more adaptable and resilient detection mechanism develops. Experimental results indicate that the Phi-4 14B model outperformed other tested models when properly fine tuned achieving a 98% accuracy rate with only a few thousand training samples. This research highlights the potential of LLMs in cybersecurity applications, specifically in mitigating domain-based deception tactics, and provides insights into optimizing machine learning strategies for threat detection.



## **35. Single Image Unlearning: Efficient Machine Unlearning in Multimodal Large Language Models**

cs.CV

**SubmitDate**: 2025-03-28    [abs](http://arxiv.org/abs/2405.12523v3) [paper-pdf](http://arxiv.org/pdf/2405.12523v3)

**Authors**: Jiaqi Li, Qianshan Wei, Chuanyi Zhang, Guilin Qi, Miaozeng Du, Yongrui Chen, Sheng Bi, Fan Liu

**Abstract**: Machine unlearning empowers individuals with the `right to be forgotten' by removing their private or sensitive information encoded in machine learning models. However, it remains uncertain whether MU can be effectively applied to Multimodal Large Language Models (MLLMs), particularly in scenarios of forgetting the leaked visual data of concepts. To overcome the challenge, we propose an efficient method, Single Image Unlearning (SIU), to unlearn the visual recognition of a concept by fine-tuning a single associated image for few steps. SIU consists of two key aspects: (i) Constructing Multifaceted fine-tuning data. We introduce four targets, based on which we construct fine-tuning data for the concepts to be forgotten; (ii) Jointly training loss. To synchronously forget the visual recognition of concepts and preserve the utility of MLLMs, we fine-tune MLLMs through a novel Dual Masked KL-divergence Loss combined with Cross Entropy loss. Alongside our method, we establish MMUBench, a new benchmark for MU in MLLMs and introduce a collection of metrics for its evaluation. Experimental results on MMUBench show that SIU completely surpasses the performance of existing methods. Furthermore, we surprisingly find that SIU can avoid invasive membership inference attacks and jailbreak attacks. To the best of our knowledge, we are the first to explore MU in MLLMs. We will release the code and benchmark in the near future.



## **36. AnyAttack: Towards Large-scale Self-supervised Adversarial Attacks on Vision-language Models**

cs.LG

CVPR 2025

**SubmitDate**: 2025-03-28    [abs](http://arxiv.org/abs/2410.05346v3) [paper-pdf](http://arxiv.org/pdf/2410.05346v3)

**Authors**: Jiaming Zhang, Junhong Ye, Xingjun Ma, Yige Li, Yunfan Yang, Yunhao Chen, Jitao Sang, Dit-Yan Yeung

**Abstract**: Due to their multimodal capabilities, Vision-Language Models (VLMs) have found numerous impactful applications in real-world scenarios. However, recent studies have revealed that VLMs are vulnerable to image-based adversarial attacks. Traditional targeted adversarial attacks require specific targets and labels, limiting their real-world impact.We present AnyAttack, a self-supervised framework that transcends the limitations of conventional attacks through a novel foundation model approach. By pre-training on the massive LAION-400M dataset without label supervision, AnyAttack achieves unprecedented flexibility - enabling any image to be transformed into an attack vector targeting any desired output across different VLMs.This approach fundamentally changes the threat landscape, making adversarial capabilities accessible at an unprecedented scale. Our extensive validation across five open-source VLMs (CLIP, BLIP, BLIP2, InstructBLIP, and MiniGPT-4) demonstrates AnyAttack's effectiveness across diverse multimodal tasks. Most concerning, AnyAttack seamlessly transfers to commercial systems including Google Gemini, Claude Sonnet, Microsoft Copilot and OpenAI GPT, revealing a systemic vulnerability requiring immediate attention.



## **37. Foot-In-The-Door: A Multi-turn Jailbreak for LLMs**

cs.CL

19 pages, 8 figures

**SubmitDate**: 2025-03-28    [abs](http://arxiv.org/abs/2502.19820v3) [paper-pdf](http://arxiv.org/pdf/2502.19820v3)

**Authors**: Zixuan Weng, Xiaolong Jin, Jinyuan Jia, Xiangyu Zhang

**Abstract**: Ensuring AI safety is crucial as large language models become increasingly integrated into real-world applications. A key challenge is jailbreak, where adversarial prompts bypass built-in safeguards to elicit harmful disallowed outputs. Inspired by psychological foot-in-the-door principles, we introduce FITD,a novel multi-turn jailbreak method that leverages the phenomenon where minor initial commitments lower resistance to more significant or more unethical transgressions. Our approach progressively escalates the malicious intent of user queries through intermediate bridge prompts and aligns the model's response by itself to induce toxic responses. Extensive experimental results on two jailbreak benchmarks demonstrate that FITD achieves an average attack success rate of 94% across seven widely used models, outperforming existing state-of-the-art methods. Additionally, we provide an in-depth analysis of LLM self-corruption, highlighting vulnerabilities in current alignment strategies and emphasizing the risks inherent in multi-turn interactions. The code is available at https://github.com/Jinxiaolong1129/Foot-in-the-door-Jailbreak.



## **38. Debate-Driven Multi-Agent LLMs for Phishing Email Detection**

cs.MA

Accepted to the 13th International Symposium on Digital Forensics and  Security (ISDFS 2025)

**SubmitDate**: 2025-03-27    [abs](http://arxiv.org/abs/2503.22038v1) [paper-pdf](http://arxiv.org/pdf/2503.22038v1)

**Authors**: Ngoc Tuong Vy Nguyen, Felix D Childress, Yunting Yin

**Abstract**: Phishing attacks remain a critical cybersecurity threat. Attackers constantly refine their methods, making phishing emails harder to detect. Traditional detection methods, including rule-based systems and supervised machine learning models, either rely on predefined patterns like blacklists, which can be bypassed with slight modifications, or require large datasets for training and still can generate false positives and false negatives. In this work, we propose a multi-agent large language model (LLM) prompting technique that simulates debates among agents to detect whether the content presented on an email is phishing. Our approach uses two LLM agents to present arguments for or against the classification task, with a judge agent adjudicating the final verdict based on the quality of reasoning provided. This debate mechanism enables the models to critically analyze contextual cue and deceptive patterns in text, which leads to improved classification accuracy. The proposed framework is evaluated on multiple phishing email datasets and demonstrate that mixed-agent configurations consistently outperform homogeneous configurations. Results also show that the debate structure itself is sufficient to yield accurate decisions without extra prompting strategies.



## **39. Intelligent IoT Attack Detection Design via ODLLM with Feature Ranking-based Knowledge Base**

cs.CR

**SubmitDate**: 2025-03-27    [abs](http://arxiv.org/abs/2503.21674v1) [paper-pdf](http://arxiv.org/pdf/2503.21674v1)

**Authors**: Satvik Verma, Qun Wang, E. Wes Bethel

**Abstract**: The widespread adoption of Internet of Things (IoT) devices has introduced significant cybersecurity challenges, particularly with the increasing frequency and sophistication of Distributed Denial of Service (DDoS) attacks. Traditional machine learning (ML) techniques often fall short in detecting such attacks due to the complexity of blended and evolving patterns. To address this, we propose a novel framework leveraging On-Device Large Language Models (ODLLMs) augmented with fine-tuning and knowledge base (KB) integration for intelligent IoT network attack detection. By implementing feature ranking techniques and constructing both long and short KBs tailored to model capacities, the proposed framework ensures efficient and accurate detection of DDoS attacks while overcoming computational and privacy limitations. Simulation results demonstrate that the optimized framework achieves superior accuracy across diverse attack types, especially when using compact models in edge computing environments. This work provides a scalable and secure solution for real-time IoT security, advancing the applicability of edge intelligence in cybersecurity.



## **40. CleanGen: Mitigating Backdoor Attacks for Generation Tasks in Large Language Models**

cs.AI

This paper is presented at EMNLP 2024

**SubmitDate**: 2025-03-27    [abs](http://arxiv.org/abs/2406.12257v3) [paper-pdf](http://arxiv.org/pdf/2406.12257v3)

**Authors**: Yuetai Li, Zhangchen Xu, Fengqing Jiang, Luyao Niu, Dinuka Sahabandu, Bhaskar Ramasubramanian, Radha Poovendran

**Abstract**: The remarkable performance of large language models (LLMs) in generation tasks has enabled practitioners to leverage publicly available models to power custom applications, such as chatbots and virtual assistants. However, the data used to train or fine-tune these LLMs is often undisclosed, allowing an attacker to compromise the data and inject backdoors into the models. In this paper, we develop a novel inference time defense, named CLEANGEN, to mitigate backdoor attacks for generation tasks in LLMs. CLEANGEN is a lightweight and effective decoding strategy that is compatible with the state-of-the-art (SOTA) LLMs. Our insight behind CLEANGEN is that compared to other LLMs, backdoored LLMs assign significantly higher probabilities to tokens representing the attacker-desired contents. These discrepancies in token probabilities enable CLEANGEN to identify suspicious tokens favored by the attacker and replace them with tokens generated by another LLM that is not compromised by the same attacker, thereby avoiding generation of attacker-desired content. We evaluate CLEANGEN against five SOTA backdoor attacks. Our results show that CLEANGEN achieves lower attack success rates (ASR) compared to five SOTA baseline defenses for all five backdoor attacks. Moreover, LLMs deploying CLEANGEN maintain helpfulness in their responses when serving benign user queries with minimal added computational overhead.



## **41. Malicious and Unintentional Disclosure Risks in Large Language Models for Code Generation**

cs.CR

The 3rd International Workshop on Mining Software Repositories  Applications for Privacy and Security (MSR4P&S), co-located with SANER 2025

**SubmitDate**: 2025-03-27    [abs](http://arxiv.org/abs/2503.22760v1) [paper-pdf](http://arxiv.org/pdf/2503.22760v1)

**Authors**: Rafiqul Rabin, Sean McGregor, Nick Judd

**Abstract**: This paper explores the risk that a large language model (LLM) trained for code generation on data mined from software repositories will generate content that discloses sensitive information included in its training data. We decompose this risk, known in the literature as ``unintended memorization,'' into two components: unintentional disclosure (where an LLM presents secrets to users without the user seeking them out) and malicious disclosure (where an LLM presents secrets to an attacker equipped with partial knowledge of the training data). We observe that while existing work mostly anticipates malicious disclosure, unintentional disclosure is also a concern. We describe methods to assess unintentional and malicious disclosure risks side-by-side across different releases of training datasets and models. We demonstrate these methods through an independent assessment of the Open Language Model (OLMo) family of models and its Dolma training datasets. Our results show, first, that changes in data source and processing are associated with substantial changes in unintended memorization risk; second, that the same set of operational changes may increase one risk while mitigating another; and, third, that the risk of disclosing sensitive information varies not only by prompt strategies or test datasets but also by the types of sensitive information. These contributions rely on data mining to enable greater privacy and security testing required for the LLM training data supply chain.



## **42. Data Poisoning in Deep Learning: A Survey**

cs.CR

**SubmitDate**: 2025-03-27    [abs](http://arxiv.org/abs/2503.22759v1) [paper-pdf](http://arxiv.org/pdf/2503.22759v1)

**Authors**: Pinlong Zhao, Weiyao Zhu, Pengfei Jiao, Di Gao, Ou Wu

**Abstract**: Deep learning has become a cornerstone of modern artificial intelligence, enabling transformative applications across a wide range of domains. As the core element of deep learning, the quality and security of training data critically influence model performance and reliability. However, during the training process, deep learning models face the significant threat of data poisoning, where attackers introduce maliciously manipulated training data to degrade model accuracy or lead to anomalous behavior. While existing surveys provide valuable insights into data poisoning, they generally adopt a broad perspective, encompassing both attacks and defenses, but lack a dedicated, in-depth analysis of poisoning attacks specifically in deep learning. In this survey, we bridge this gap by presenting a comprehensive and targeted review of data poisoning in deep learning. First, this survey categorizes data poisoning attacks across multiple perspectives, providing an in-depth analysis of their characteristics and underlying design princinples. Second, the discussion is extended to the emerging area of data poisoning in large language models(LLMs). Finally, we explore critical open challenges in the field and propose potential research directions to advance the field further. To support further exploration, an up-to-date repository of resources on data poisoning in deep learning is available at https://github.com/Pinlong-Zhao/Data-Poisoning.



## **43. Harnessing Chain-of-Thought Metadata for Task Routing and Adversarial Prompt Detection**

cs.CL

**SubmitDate**: 2025-03-27    [abs](http://arxiv.org/abs/2503.21464v1) [paper-pdf](http://arxiv.org/pdf/2503.21464v1)

**Authors**: Ryan Marinelli, Josef Pichlmeier, Tamas Bisztray

**Abstract**: In this work, we propose a metric called Number of Thoughts (NofT) to determine the difficulty of tasks pre-prompting and support Large Language Models (LLMs) in production contexts. By setting thresholds based on the number of thoughts, this metric can discern the difficulty of prompts and support more effective prompt routing. A 2% decrease in latency is achieved when routing prompts from the MathInstruct dataset through quantized, distilled versions of Deepseek with 1.7 billion, 7 billion, and 14 billion parameters. Moreover, this metric can be used to detect adversarial prompts used in prompt injection attacks with high efficacy. The Number of Thoughts can inform a classifier that achieves 95% accuracy in adversarial prompt detection. Our experiments ad datasets used are available on our GitHub page: https://github.com/rymarinelli/Number_Of_Thoughts/tree/main.



## **44. Tricking Retrievers with Influential Tokens: An Efficient Black-Box Corpus Poisoning Attack**

cs.LG

Accepted to NAACL 2025 Main Track

**SubmitDate**: 2025-03-27    [abs](http://arxiv.org/abs/2503.21315v1) [paper-pdf](http://arxiv.org/pdf/2503.21315v1)

**Authors**: Cheng Wang, Yiwei Wang, Yujun Cai, Bryan Hooi

**Abstract**: Retrieval-augmented generation (RAG) systems enhance large language models by incorporating external knowledge, addressing issues like outdated internal knowledge and hallucination. However, their reliance on external knowledge bases makes them vulnerable to corpus poisoning attacks, where adversarial passages can be injected to manipulate retrieval results. Existing methods for crafting such passages, such as random token replacement or training inversion models, are often slow and computationally expensive, requiring either access to retriever's gradients or large computational resources. To address these limitations, we propose Dynamic Importance-Guided Genetic Algorithm (DIGA), an efficient black-box method that leverages two key properties of retrievers: insensitivity to token order and bias towards influential tokens. By focusing on these characteristics, DIGA dynamically adjusts its genetic operations to generate effective adversarial passages with significantly reduced time and memory usage. Our experimental evaluation shows that DIGA achieves superior efficiency and scalability compared to existing methods, while maintaining comparable or better attack success rates across multiple datasets.



## **45. M-LLM Based Video Frame Selection for Efficient Video Understanding**

cs.CV

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2502.19680v2) [paper-pdf](http://arxiv.org/pdf/2502.19680v2)

**Authors**: Kai Hu, Feng Gao, Xiaohan Nie, Peng Zhou, Son Tran, Tal Neiman, Lingyun Wang, Mubarak Shah, Raffay Hamid, Bing Yin, Trishul Chilimbi

**Abstract**: Recent advances in Multi-Modal Large Language Models (M-LLMs) show promising results in video reasoning. Popular Multi-Modal Large Language Model (M-LLM) frameworks usually apply naive uniform sampling to reduce the number of video frames that are fed into an M-LLM, particularly for long context videos. However, it could lose crucial context in certain periods of a video, so that the downstream M-LLM may not have sufficient visual information to answer a question. To attack this pain point, we propose a light-weight M-LLM -based frame selection method that adaptively select frames that are more relevant to users' queries. In order to train the proposed frame selector, we introduce two supervision signals (i) Spatial signal, where single frame importance score by prompting a M-LLM; (ii) Temporal signal, in which multiple frames selection by prompting Large Language Model (LLM) using the captions of all frame candidates. The selected frames are then digested by a frozen downstream video M-LLM for visual reasoning and question answering. Empirical results show that the proposed M-LLM video frame selector improves the performances various downstream video Large Language Model (video-LLM) across medium (ActivityNet, NExT-QA) and long (EgoSchema, LongVideoBench) context video question answering benchmarks.



## **46. Iterative Prompting with Persuasion Skills in Jailbreaking Large Language Models**

cs.CL

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2503.20320v1) [paper-pdf](http://arxiv.org/pdf/2503.20320v1)

**Authors**: Shih-Wen Ke, Guan-Yu Lai, Guo-Lin Fang, Hsi-Yuan Kao

**Abstract**: Large language models (LLMs) are designed to align with human values in their responses. This study exploits LLMs with an iterative prompting technique where each prompt is systematically modified and refined across multiple iterations to enhance its effectiveness in jailbreaking attacks progressively. This technique involves analyzing the response patterns of LLMs, including GPT-3.5, GPT-4, LLaMa2, Vicuna, and ChatGLM, allowing us to adjust and optimize prompts to evade the LLMs' ethical and security constraints. Persuasion strategies enhance prompt effectiveness while maintaining consistency with malicious intent. Our results show that the attack success rates (ASR) increase as the attacking prompts become more refined with the highest ASR of 90% for GPT4 and ChatGLM and the lowest ASR of 68% for LLaMa2. Our technique outperforms baseline techniques (PAIR and PAP) in ASR and shows comparable performance with GCG and ArtPrompt.



## **47. Stealthy Backdoor Attack in Self-Supervised Learning Vision Encoders for Large Vision Language Models**

cs.CV

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2502.18290v3) [paper-pdf](http://arxiv.org/pdf/2502.18290v3)

**Authors**: Zhaoyi Liu, Huan Zhang

**Abstract**: Self-supervised learning (SSL) vision encoders learn high-quality image representations and thus have become a vital part of developing vision modality of large vision language models (LVLMs). Due to the high cost of training such encoders, pre-trained encoders are widely shared and deployed into many LVLMs, which are security-critical or bear societal significance. Under this practical scenario, we reveal a new backdoor threat that significant visual hallucinations can be induced into these LVLMs by merely compromising vision encoders. Because of the sharing and reuse of these encoders, many downstream LVLMs may inherit backdoor behaviors from encoders, leading to widespread backdoors. In this work, we propose BadVision, the first method to exploit this vulnerability in SSL vision encoders for LVLMs with novel trigger optimization and backdoor learning techniques. We evaluate BadVision on two types of SSL encoders and LVLMs across eight benchmarks. We show that BadVision effectively drives the LVLMs to attacker-chosen hallucination with over 99% attack success rate, causing a 77.6% relative visual understanding error while maintaining the stealthiness. SoTA backdoor detection methods cannot detect our attack effectively.



## **48. TeleLoRA: Teleporting Model-Specific Alignment Across LLMs**

cs.LG

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2503.20228v1) [paper-pdf](http://arxiv.org/pdf/2503.20228v1)

**Authors**: Xiao Lin, Manoj Acharya, Anirban Roy, Susmit Jha

**Abstract**: Mitigating Trojans in Large Language Models (LLMs) is one of many tasks where alignment data is LLM specific, as different LLMs have different Trojan triggers and trigger behaviors to be removed. In this paper, we introduce TeleLoRA (Teleporting Low-Rank Adaptation), a novel framework that synergizes model-specific alignment data across multiple LLMs to enable zero-shot Trojan mitigation on unseen LLMs without alignment data. TeleLoRA learns a unified generator of LoRA adapter weights by leveraging local activation information across multiple LLMs. This generator is designed to be permutation symmetric to generalize across models with different architectures and sizes. We optimize the model design for memory efficiency, making it feasible to learn with large-scale LLMs with minimal computational resources. Experiments on LLM Trojan mitigation benchmarks demonstrate that TeleLoRA effectively reduces attack success rates while preserving the benign performance of the models.



## **49. Playing the Fool: Jailbreaking LLMs and Multimodal LLMs with Out-of-Distribution Strategy**

cs.CR

Accepted at CVPR2025

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2503.20823v1) [paper-pdf](http://arxiv.org/pdf/2503.20823v1)

**Authors**: Joonhyun Jeong, Seyun Bae, Yeonsung Jung, Jaeryong Hwang, Eunho Yang

**Abstract**: Despite the remarkable versatility of Large Language Models (LLMs) and Multimodal LLMs (MLLMs) to generalize across both language and vision tasks, LLMs and MLLMs have shown vulnerability to jailbreaking, generating textual outputs that undermine safety, ethical, and bias standards when exposed to harmful or sensitive inputs. With the recent advancement of safety alignment via preference-tuning from human feedback, LLMs and MLLMs have been equipped with safety guardrails to yield safe, ethical, and fair responses with regard to harmful inputs. However, despite the significance of safety alignment, research on the vulnerabilities remains largely underexplored. In this paper, we investigate the unexplored vulnerability of the safety alignment, examining its ability to consistently provide safety guarantees for out-of-distribution(OOD)-ifying harmful inputs that may fall outside the aligned data distribution. Our key observation is that OOD-ifying the vanilla harmful inputs highly increases the uncertainty of the model to discern the malicious intent within the input, leading to a higher chance of being jailbroken. Exploiting this vulnerability, we propose JOOD, a new Jailbreak framework via OOD-ifying inputs beyond the safety alignment. We explore various off-the-shelf visual and textual transformation techniques for OOD-ifying the harmful inputs. Notably, we observe that even simple mixing-based techniques such as image mixup prove highly effective in increasing the uncertainty of the model, thereby facilitating the bypass of the safety alignment. Experiments across diverse jailbreak scenarios demonstrate that JOOD effectively jailbreaks recent proprietary LLMs and MLLMs such as GPT-4 and o1 with high attack success rate, which previous attack approaches have consistently struggled to jailbreak. Code is available at https://github.com/naver-ai/JOOD.



## **50. Knowledge Transfer from LLMs to Provenance Analysis: A Semantic-Augmented Method for APT Detection**

cs.CR

**SubmitDate**: 2025-03-25    [abs](http://arxiv.org/abs/2503.18316v2) [paper-pdf](http://arxiv.org/pdf/2503.18316v2)

**Authors**: Fei Zuo, Junghwan Rhee, Yung Ryn Choe

**Abstract**: Advanced Persistent Threats (APTs) have caused significant losses across a wide range of sectors, including the theft of sensitive data and harm to system integrity. As attack techniques grow increasingly sophisticated and stealthy, the arms race between cyber defenders and attackers continues to intensify. The revolutionary impact of Large Language Models (LLMs) has opened up numerous opportunities in various fields, including cybersecurity. An intriguing question arises: can the extensive knowledge embedded in LLMs be harnessed for provenance analysis and play a positive role in identifying previously unknown malicious events? To seek a deeper understanding of this issue, we propose a new strategy for taking advantage of LLMs in provenance-based threat detection. In our design, the state-of-the-art LLM offers additional details in provenance data interpretation, leveraging their knowledge of system calls, software identity, and high-level understanding of application execution context. The advanced contextualized embedding capability is further utilized to capture the rich semantics of event descriptions. We comprehensively examine the quality of the resulting embeddings, and it turns out that they offer promising avenues. Subsequently, machine learning models built upon these embeddings demonstrated outstanding performance on real-world data. In our evaluation, supervised threat detection achieves a precision of 99.0%, and semi-supervised anomaly detection attains a precision of 96.9%.



