
from workflow_manager import WorkflowManager, Job

# ... existing code ...

# Workflow API Models
class WorkflowJobCreateRequest(BaseModel):
    project_path: str
    batch_size: int
    annotator: str
    reviewer: str
    source: str = "unassigned"

class WorkflowJobActionRequest(BaseModel):
    project_path: str

class WorkflowAnnotationSaveRequest(BaseModel):
    project_path: str
    image_name: str
    content: str

class WorkflowImageActionRequest(BaseModel):
    project_path: str
    image_name: str
    comment: Optional[str] = None

# Workflow Endpoints

@app.get("/api/workflow/unassigned")
def get_workflow_unassigned_count(project_path: str):
    try:
        wm = WorkflowManager(project_path)
        count = wm.get_unassigned_count()
        return {"count": count}
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/workflow/jobs")
def get_workflow_jobs(project_path: str):
    try:
        wm = WorkflowManager(project_path)
        return wm.get_jobs()
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/workflow/jobs")
def create_workflow_job(request: WorkflowJobCreateRequest):
    try:
        wm = WorkflowManager(request.project_path)
        return wm.create_job(
            request.batch_size,
            request.annotator,
            request.reviewer,
            request.source
        )
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/workflow/job/{job_id}")
def get_workflow_job(job_id: str, project_path: str):
    try:
        wm = WorkflowManager(project_path)
        job = wm.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/workflow/job/{job_id}/submit")
def submit_workflow_job(job_id: str, request: WorkflowJobActionRequest):
    try:
        wm = WorkflowManager(request.project_path)
        wm.submit_job(job_id)
        return {"status": "success"}
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/workflow/job/{job_id}/return")
def return_workflow_job(job_id: str, request: WorkflowJobActionRequest):
    try:
        wm = WorkflowManager(request.project_path)
        wm.return_job(job_id)
        return {"status": "success"}
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/workflow/job/{job_id}/annotation")
def save_workflow_annotation(job_id: str, request: WorkflowAnnotationSaveRequest):
    try:
        wm = WorkflowManager(request.project_path)
        wm.save_annotation(job_id, request.image_name, request.content)
        return {"status": "success"}
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/workflow/job/{job_id}/approve")
def approve_workflow_image(job_id: str, request: WorkflowImageActionRequest):
    try:
        wm = WorkflowManager(request.project_path)
        wm.approve_image(job_id, request.image_name)
        return {"status": "success"}
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/workflow/job/{job_id}/reject")
def reject_workflow_image(job_id: str, request: WorkflowImageActionRequest):
    try:
        wm = WorkflowManager(request.project_path)
        wm.reject_image(job_id, request.image_name, request.comment or "")
        return {"status": "success"}
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))
