# Design Note

## 1. Monitoring & Alerting

### Metrics to Track

- **Citation verification rate** - Should be > 80%
- **Abstention rate on answerable questions** - Should be < 20%
- **Non-JSON responses** - Should be < 5%
- **Response time** - Should be < 10 seconds

### When to Alert

**Immediate (Call someone)**
- Citation rate drops below 70%
- More than 10% of responses are not JSON
- System stops answering (index corruption)

**Warning (Check daily)**
- Citation rate between 70-80%
- More abstentions than usual
- Slower response times

## 2. Document Updates

### When a Document Changes

1. **Re-ingest** the updated document
2. **Flag** all answers that used that document as "stale"
3. **Re-run** those questions through the system
4. **Review** answers that changed significantly

### Preventing Stale Answers

- Track which document version each answer used
- Re-validate answers weekly
- Clear cache when documents update

## 3. When to Refuse & Hand Off

### System Should Refuse If:

1. **Low confidence** - Not enough information found
2. **Conflicting docs** - Documents disagree on the answer
3. **Sensitive topic** - Questions about personal data
4. **Too complex** - Requires human judgment or legal interpretation

### Human Review Priority

| Priority | Type | Response Time |
|----------|------|---------------|
| 1 | Legal interpretation | 24 hours |
| 2 | Conflicting documents | 48 hours |
| 3 | Low confidence | 72 hours |

## 4. Cost

### Current (Local)
- **Cost per question**: $0
- **Monthly (10,000 questions)**: $0

### Cloud Alternative
- **Cost per question**: ~$0.001
- **Monthly (10,000 questions)**: ~$10

### Cost Savings

| Strategy | Savings |
|----------|---------|
| Cache common questions | 80% |
| Batch processing | 30% |
| Use smaller model for simple questions | 60% |

## 5. Scaling

### Limitations

- Local LLM is slow (2-3 seconds per question)
- FAISS index uses RAM
- Single machine can fail

### Solution

```
User → API Gateway → Load Balancer → RAG Service (3 replicas)
                                    ↓
                          Embedding Cache → FAISS Index
                                    ↓
                          LLM Service (GPU)
```

### Cloud Cost Estimate

| Service | Monthly Cost |
|---------|--------------|
| Compute (3 servers) | $60 |
| GPU | $120 |
| Storage | $10 |
| LLM API | $100 |
| **Total** | **~$290** |

### Priority Improvements

1. **Now**: Add caching
2. **Week 1**: Add monitoring
3. **Week 2**: Add human review
4. **Week 3**: Add document versioning
5. **Week 4**: Scale to multiple servers

## 6. Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Document updates break answers | Medium | Version tracking + re-validation |
| LLM makes up answers | High | Citation validation + abstention |
| System goes down | Medium | Load balancer + replicas |
| Data breach | Very High | No PII stored |
| Wrong answers | High | Human review queue |

## Summary

The system can handle 10,000 questions/month for $0 (local) or ~$10 (cloud). Key priorities:

1. **Add caching** - saves 80% cost
2. **Monitor** - catch issues early
3. **Human review** - handle edge cases
4. **Version tracking** - handle document updates
5. **Scale** - handle more users

**Biggest risk**: LLM making up answers (hallucination). Fixed by:
- Requiring verbatim citations
- Abstaining when not sure
- Human review for complex cases
