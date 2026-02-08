import os
import json
import shutil
import random
from pathlib import Path
from typing import List, Dict, Optional, Union
from pydantic import BaseModel
from datetime import datetime

class JobImageStatus(BaseModel):
    status: str = "pending"  # pending, done (approved), rejected
    comments: List[str] = []

class Job(BaseModel):
    id: str
    name: str # e.g. "Job #1 (Annotating)"
    status: str # annotating, review, done
    annotator: str
    reviewer: str
    created_at: str
    updated_at: str
    stage: str # "annotation" (new labels) or "review" (fixing dataset)
    parent_id: Optional[str] = None
    images: Dict[str, JobImageStatus] = {}

class WorkflowManager:
    JOBS_FILENAME = "jobs.json"
    STATE_FILENAME = "workflow_state.json"
    USERS_FILENAME = "users.json"
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.jobs_file = self.project_path / self.JOBS_FILENAME
        self.state_file = self.project_path / self.STATE_FILENAME
        self.users_file = self.project_path / self.USERS_FILENAME
        self.jobs_dir = self.project_path / "jobs"
        self.ensure_structure()
        self.migrate_to_json_state()
        
    def ensure_structure(self):
        # self.jobs_dir.mkdir(exist_ok=True)
        if not self.jobs_file.exists():
            self.save_jobs({"next_job_id": 1, "jobs": {}})
        if not self.users_file.exists():
            self.save_users({"users": ["admin"]})
            
    def load_users(self) -> Dict:
        if not self.users_file.exists():
            return {"users": ["admin"]}
        try:
            with open(self.users_file, 'r') as f:
                return json.load(f)
        except:
            return {"users": ["admin"]}

    def save_users(self, data: Dict):
        with open(self.users_file, 'w') as f:
            json.dump(data, f, indent=4)

    def get_users(self) -> List[str]:
        data = self.load_users()
        users = data.get("users", [])
        if "admin" not in users:
            users.insert(0, "admin")
            self.save_users({"users": users})
        return users

    def register_user(self, name: str) -> List[str]:
        name = name.strip()
        if not name:
            return self.get_users()
        data = self.load_users()
        users = data.get("users", [])
        if name not in users:
            users.append(name)
            self.save_users({"users": users})
        return users

    def load_state(self) -> Dict:
        if not self.state_file.exists():
            return {"images": {}}
        with open(self.state_file, 'r') as f:
            return json.load(f)

    def save_state(self, data: Dict):
        with open(self.state_file, 'w') as f:
            json.dump(data, f, indent=4)

    def migrate_to_json_state(self):
        """Migrates from folder-based state to JSON-based state."""
        if self.state_file.exists():
            return # Already migrated
            
        # Prevent race conditions with a simple lock file or check again
        try:
             # Basic double-check pattern
             if self.state_file.exists(): return
             
             print("Migrating to JSON-based state...")
             # Create empty state first to "claim" migration
             state_data = {"images": {}}
             # self.save_state(state_data) # Don't save yet, might be incomplete. 
             # But if we don't save, other processes might start.
             # Ideally we use a .lock file, but let's just write a 'pending' state?
             
             # Load config for dir names
             project_config_path = self.project_path / "project_config.json"
             processed_dir_name = "images_processed"
             annotations_dir_name = "annotations"
             draft_dir_name = "annotations_draft"
             
             if project_config_path.exists():
                  try:
                     with open(project_config_path) as f:
                         cfg = json.load(f)
                         processed_dir_name = cfg.get("dirs", {}).get("processed", processed_dir_name)
                         annotations_dir_name = cfg.get("dirs", {}).get("annotations", annotations_dir_name)
                  except: pass
                  
             processed_dir = self.project_path / processed_dir_name
             annotations_dir = self.project_path / annotations_dir_name
             draft_dir = self.project_path / draft_dir_name
             
             # Ensure central annotations dir exists
             annotations_dir.mkdir(parents=True, exist_ok=True)
             
             # 1. Register all processed images as Unassigned initially
             if processed_dir.exists():
                 for f in processed_dir.iterdir():
                     if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                         state_data["images"][f.name] = {
                             "status": "unassigned",
                             "job_id": None,
                             "last_updated": datetime.now().isoformat()
                         }
                         
             # 2. Mark Dataset images (in annotations/)
             if annotations_dir.exists():
                 for f in annotations_dir.glob("*.txt"):
                     stem = f.stem
                     # Find image key with same stem
                     matched_img = None
                     for img_name in state_data["images"]:
                         if Path(img_name).stem == stem:
                             matched_img = img_name
                             break
                     
                     if matched_img:
                         state_data["images"][matched_img]["status"] = "dataset"
                         
             # 3. Mark Job images and move files
             jobs_data = self.load_jobs_data()
             for jid, job in jobs_data.get("jobs", {}).items():
                 # If job not done, images are assigned
                 if job.get("status") != "done":
                     for img_name in job.get("images", {}):
                          if img_name in state_data["images"]:
                              state_data["images"][img_name]["status"] = "assigned"
                              state_data["images"][img_name]["job_id"] = jid
                              
                 # Move annotations from job folder to central
                 job_ann_dir = self.jobs_dir / str(jid) / "annotations"
                 if job_ann_dir.exists():
                     for f in job_ann_dir.glob("*.txt"):
                         if not (annotations_dir / f.name).exists(): # Don't overwrite dataset
                              shutil.move(str(f), str(annotations_dir / f.name))
                     # Remove job annotations dir? Yes.
                     shutil.rmtree(str(job_ann_dir))
                     
             # 4. Handle Drafts (move to central, mark unassigned)
             if draft_dir.exists():
                 for f in draft_dir.glob("*.txt"):
                      if not (annotations_dir / f.name).exists():
                          shutil.move(str(f), str(annotations_dir / f.name))
                 shutil.rmtree(str(draft_dir))
                 
             self.save_state(state_data)
             print("Migration to JSON state complete.")
             
        except Exception as e:
            print(f"Error during migration: {e}")
            # Ensure we don't leave broken state? 
            # Ideally retry or log.

    def _get_annotations_dir(self) -> Path:
        """Helper to resolve the current annotations directory from config."""
        annotations_dir = self.project_path / "annotations"
        project_config_path = self.project_path / "project_config.json"
        if project_config_path.exists():
            try:
                with open(project_config_path) as f:
                    cfg = json.load(f)
                    annotations_dir = self.project_path / cfg.get("dirs", {}).get("annotations", "annotations")
            except: pass
        return annotations_dir

    def load_jobs_data(self) -> Dict:
        if not self.jobs_file.exists():
             return {"next_job_id": 1, "jobs": {}}
        with open(self.jobs_file, 'r') as f:
            return json.load(f)

    def save_jobs(self, data: Dict):
        with open(self.jobs_file, 'w') as f:
            json.dump(data, f, indent=4)
            
    def sync_images(self, state_data: Dict) -> bool:
        """Scans processed directory and adds new images to state. 
           Also scans annotations directory and adds 'dataset' status if missing."""
        project_config_path = self.project_path / "project_config.json"
        processed_dir_name = "images_processed"
        annotations_dir_name = "annotations"
        if project_config_path.exists():
             try:
                with open(project_config_path) as f:
                    cfg = json.load(f)
                    processed_dir_name = cfg.get("dirs", {}).get("processed", processed_dir_name)
                    annotations_dir_name = cfg.get("dirs", {}).get("annotations", annotations_dir_name)
             except: pass
             
        processed_dir = self.project_path / processed_dir_name
        annotations_dir = self.project_path / annotations_dir_name
        
        updated = False
        
        # 1. Sync Images (Files -> State)
        if processed_dir.exists():
            for f in processed_dir.iterdir():
                if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    if f.name not in state_data["images"]:
                        state_data["images"][f.name] = {
                            "status": "unassigned",
                            "job_id": None,
                            "last_updated": datetime.now().isoformat()
                        }
                        updated = True
                        
        # 2. Sync Annotations (Files -> State)
        return updated
            
    def get_jobs(self) -> List[Dict]:
        data = self.load_jobs_data()
        annotations_dir = self._get_annotations_dir()
        existing_annots = set()
        if annotations_dir.exists():
            existing_annots = {f.stem for f in annotations_dir.glob("*.txt")}

        # Convert dict jobs to list for frontend, sorted by ID desc
        jobs_list = []
        for jid, jdata in data["jobs"].items():
            jdata["id"] = jid
            # Add stats
            total = len(jdata.get("images", {}))
            
            # Smart stats: Count as 'done' if status is 'done' OR if file exists on disk (but only if in annotating stage)
            done = 0
            pending = 0
            rejected = sum(1 for i in jdata.get("images", {}).values() if i["status"] == "rejected")
            job_status = jdata.get("status", "annotating")
            
            for img, info in jdata.get("images", {}).items():
                is_done_in_state = info["status"] == "done"
                has_file = Path(img).stem in existing_annots
                
                if is_done_in_state:
                    done += 1
                elif info["status"] == "pending" and has_file and job_status == "annotating":
                    # In annotating stage, having a file means it's "labeled" (UI uses 'done' stat for this)
                    done += 1
                elif info["status"] == "pending":
                    pending += 1
            
            jdata["stats"] = {"total": total, "done": done, "pending": pending, "rejected": rejected}
            jobs_list.append(jdata)
        
        return sorted(jobs_list, key=lambda x: int(x["id"]), reverse=True)

    def get_job(self, job_id: str) -> Optional[Dict]:
        data = self.load_jobs_data()
        job = data["jobs"].get(str(job_id))
        if job:
            job["id"] = str(job_id)
        return job

    def create_job(self, batch_size: int, annotator: str, reviewer: str, source: str = "unassigned") -> Dict:
        data = self.load_jobs_data()
        job_id = str(data["next_job_id"])
        
        # 1. Select Images
        selected_images = []
        
        project_config_path = self.project_path / "project_config.json"
        processed_dir_name = "images_processed" # Default
        annotations_dir_name = "annotations" # Default
        
        if project_config_path.exists():
             try:
                with open(project_config_path) as f:
                    cfg = json.load(f)
                    processed_dir_name = cfg.get("dirs", {}).get("processed", processed_dir_name)
                    annotations_dir_name = cfg.get("dirs", {}).get("annotations", annotations_dir_name)
             except: pass
             
        processed_dir = self.project_path / processed_dir_name
        annotations_dir = self.project_path / annotations_dir_name
        
        # Helper: Get all images currently in jobs to exclude them
        in_job_images = set()
        for j in data["jobs"].values():
            if j["status"] != "done": # Only exclude active job images? 
                # Actually, if source is 'unassigned', we exclude ALL images ever assigned? 
                # Or just active? 
                # Ideally, if a job is done, the images are in 'dataset' state.
                # If we want to re-review them, we can. 
                # But 'unassigned' specifically means 'not in any job' AND 'not in dataset'.
                # But wait, if they are in 'done' job, they are in dataset.
                # So 'unassigned' = processed images - (images in ANY job)
                # This prevents duplicate assignments.
                for img, info in j["images"].items():
                    if info.get("status") != "done":
                        in_job_images.add(img)

    def unassign_job(self, job_id: str):
        data = self.load_jobs_data()
        job = data["jobs"].get(str(job_id))
        
        if job:
            # Update State: Mark images as Unassigned
            state = self.load_state()
            for img_name in job["images"]:
                if img_name in state["images"]:
                     state["images"][img_name]["status"] = "unassigned"
                     state["images"][img_name]["job_id"] = None
                     state["images"][img_name]["last_updated"] = datetime.now().isoformat()
            
            self.save_state(state)
            
            # Remove from jobs.json
            del data["jobs"][str(job_id)]
            self.save_jobs(data)
            
            # Delete the job directory (metadata)
            job_dir = self.jobs_dir / str(job_id)
            if job_dir.exists():
                shutil.rmtree(str(job_dir))

    def create_job(self, batch_size: int, annotator: str, reviewer: str, source: str = "unassigned", include_annotated: bool = False, source_job_id: Optional[str] = None) -> Dict:
        state = self.load_state()
        self.sync_images(state)
        
        data = self.load_jobs_data()
        job_id = str(data["next_job_id"])
        
        selected_images = []
        stage = "annotation"
        
        annotations_dir = self.project_path / "annotations"
        # Ensure config is respected for dir name
        project_config_path = self.project_path / "project_config.json"
        if project_config_path.exists():
             try:
                with open(project_config_path) as f:
                    cfg = json.load(f)
                    annotations_dir = self.project_path / cfg.get("dirs", {}).get("annotations", "annotations")
             except: pass

        if not reviewer:
             reviewer = annotator
             
        if source == "unassigned":
            # Select from Unassigned
            candidates = []
            for img, info in state["images"].items():
                if info["status"] == "unassigned":
                    # Check for existing annotation availability
                    has_annotation = (annotations_dir / f"{Path(img).stem}.txt").exists()
                    if not has_annotation or include_annotated:
                        candidates.append(img)
            
            candidates.sort() # consistency
            
            if len(candidates) < batch_size:
                raise ValueError(f"Not enough images. Requested {batch_size}, found {len(candidates)}.")
                
            selected_images = candidates[:batch_size]
            stage = "annotation"
            
        elif source == "dataset":
            candidates = []
            # Optionally filter by source_job_id using jobs.json history
            if source_job_id:
                source_job = data["jobs"].get(str(source_job_id))
                if source_job:
                    for img in source_job["images"]:
                        # Verify it is currently in dataset state (not re-assigned elsewhere)
                        if state["images"].get(img, {}).get("status") == "dataset":
                            candidates.append(img)
            else:
                for img, info in state["images"].items():
                    if info["status"] == "dataset":
                        candidates.append(img)
                        
            candidates.sort()
            
            if len(candidates) < batch_size:
                 raise ValueError(f"Not enough images. Requested {batch_size}, found {len(candidates)}.")
            
            selected_images = candidates[:batch_size]
            stage = "review" # Re-reviewing dataset images

        data["next_job_id"] += 1
        
        # Update State (Mark as Assigned)
        for img in selected_images:
            state["images"][img]["status"] = "assigned"
            state["images"][img]["job_id"] = job_id
            state["images"][img]["last_updated"] = datetime.now().isoformat()
            
        self.save_state(state)
        
        # Create Job Entry (No directory creation for annotations needed)
        # But we create the job folder for metadata/consistency if plugins need it?
        # Actually, let's keep the job folder for future extensibility (e.g. comments log file?)
        # But NOT 'annotations' subdir.
        # (self.jobs_dir / job_id).mkdir(parents=True, exist_ok=True)

        # Prepare images with correct status
        job_images = {}
        for img in selected_images:
             # Check if annotation exists, but only mark DONE if we are in Annotation stage (Drafts)
             # If Review stage, we want them Pending (to be reviewed)
             status = "pending"
             status = "pending"
             if stage == "annotation":
                 txt_path = annotations_dir / f"{Path(img).stem}.txt"
                 # if txt_path.exists():
                 #     status = "done"
            
             job_images[img] = JobImageStatus(status=status)

        new_job = Job(
            id=job_id,
            name=f"Job #{job_id} ({stage.title()})",
            status="annotating" if stage == "annotation" else "review",
            annotator=annotator,
            reviewer=reviewer,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            stage=stage,
            images=job_images
        )
        data["jobs"][job_id] = new_job.model_dump()
        self.save_jobs(data)
        
        return data["jobs"][job_id]

    def _find_image_for_label(self, label_path: Path, images_dir: Path) -> Optional[str]:
        # Simple heuristic
        stem = label_path.stem
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            if (images_dir / f"{stem}{ext}").exists():
                return f"{stem}{ext}"
        return None

    def save_annotation(self, job_id: str, image_name: str, content: str):
        data = self.load_jobs_data()
        job = data["jobs"].get(str(job_id))
        if not job: raise ValueError("Job not found")
        
        # Ensure project annotations dir exists
        project_config_path = self.project_path / "project_config.json"
        annotations_dir_name = "annotations"
        if project_config_path.exists():
             try:
                with open(project_config_path) as f:
                    cfg = json.load(f)
                    annotations_dir_name = cfg.get("dirs", {}).get("annotations", annotations_dir_name)
             except: pass
        
        target_dir = self.project_path / annotations_dir_name
        target_dir.mkdir(parents=True, exist_ok=True)
        
        with open(target_dir / f"{Path(image_name).stem}.txt", 'w') as f:
            f.write(content)

        if job["status"] == "annotating":
             job["images"][image_name]["status"] = "done"
             
        # Update State Registry timestamp
        state = self.load_state()
        if image_name in state["images"]:
             state["images"][image_name]["last_updated"] = datetime.now().isoformat()
        self.save_state(state)
        
        job["updated_at"] = datetime.now().isoformat()
        self.save_jobs(data)

    def submit_job(self, job_id: str):
        data = self.load_jobs_data()
        job = data["jobs"].get(str(job_id))
        if not job: raise ValueError("Job not found")
        
        # Check if all images have annotations? 
        # Optional: For now, strict check.
        # Check if file exists for all
        # for img in job["images"]:
        #      path = self.jobs_dir / str(job_id) / "annotations" / f"{Path(img).stem}.txt"
        #      if not path.exists():
        #          raise ValueError(f"Image {img} is missing annotation")
                 
        if job["status"] == "annotating":
            # Transition to Review:
            # Mark all DONE images as PENDING for the reviewer to see them
            for img, info in job["images"].items():
                if info["status"] == "done":
                    info["status"] = "pending"

        job["status"] = "review"
        job["updated_at"] = datetime.now().isoformat()
        self.save_jobs(data)

    def approve_job(self, job_id: str):
        data = self.load_jobs_data()
        job = data["jobs"].get(str(job_id))
        if not job: raise ValueError("Job not found")

        # Update State Registry
        state = self.load_state()
        
        for image_name, info in job["images"].items():
            # If not rejected, it is approved?
            # Usually only "done" images are approved? 
            # Or "pending" are auto-approved?
            # Let's say if it's NOT rejected, it's approved.
            if info["status"] != "rejected":
                # Ensure annotation exists?
                # It should, because we are using central dir.
                
                info["status"] = "done"
                
                if image_name in state["images"]:
                    state["images"][image_name]["status"] = "dataset"
                    state["images"][image_name]["last_updated"] = datetime.now().isoformat()
        
        self.save_state(state)
        
        job["status"] = "done"
        job["updated_at"] = datetime.now().isoformat()
        self.save_jobs(data)

    def return_job(self, job_id: str):
        data = self.load_jobs_data()
        job = data["jobs"].get(str(job_id))
        if not job: raise ValueError("Job not found")
        
        job["status"] = "annotating"
        job["updated_at"] = datetime.now().isoformat()
        self.save_jobs(data)

    def approve_image(self, job_id: str, image_name: str):
        data = self.load_jobs_data()
        job = data["jobs"].get(str(job_id))
        if not job: raise ValueError("Job not found")
        
        # Move annotation to main folder
        src = self.jobs_dir / str(job_id) / "annotations" / f"{Path(image_name).stem}.txt"
        
        # Ensure project annotations dir exists
        project_config_path = self.project_path / "project_config.json"
        annotations_dir_name = "annotations"
        if project_config_path.exists():
             try:
                with open(project_config_path) as f:
                    cfg = json.load(f)
                    annotations_dir_name = cfg.get("dirs", {}).get("annotations", annotations_dir_name)
             except: pass
        
        target_dir = self.project_path / annotations_dir_name
        target_dir.mkdir(parents=True, exist_ok=True)
        
        if src.exists():
            shutil.copy2(str(src), str(target_dir / src.name))
        else:
            # Maybe approval of empty annotation (empty image)?
            # Create empty file
            with open(target_dir / src.name, 'w') as f:
                pass

        job["images"][image_name]["status"] = "done"
        job["updated_at"] = datetime.now().isoformat()
        
        # user feedback: remove it from review state
        # Handled by logic: if status is done, it's not pending review.
        
        # If all done, job is done?
        all_done = all(i["status"] == "done" for i in job["images"].values())
        if all_done:
            job["status"] = "done"
            
        self.save_jobs(data)

    def get_annotation(self, job_id: str, image_name: str) -> Dict:
        # Return content and status
        data = self.load_jobs_data()
        job = data["jobs"].get(str(job_id))
        if not job: raise ValueError("Job not found")
        
        # Central annotations dir
        project_config_path = self.project_path / "project_config.json"
        annotations_dir_name = "annotations"
        if project_config_path.exists():
             try:
                with open(project_config_path) as f:
                    cfg = json.load(f)
                    annotations_dir_name = cfg.get("dirs", {}).get("annotations", annotations_dir_name)
             except: pass
             
        txt_path = self.project_path / annotations_dir_name / f"{Path(image_name).stem}.txt"
        
        content = ""
        if txt_path.exists():
            with open(txt_path, 'r') as f:
                content = f.read()
                 
        return {"content": content, "status": job["images"].get(image_name, {}).get("status", "pending")}
    def reject_image(self, job_id: str, image_name: str, comment: str):
        data = self.load_jobs_data()
        job = data["jobs"].get(str(job_id))
        if not job: raise ValueError("Job not found")
        
        # User feedback: Move rejected image to a "Correction Job" in Annotating state
        # NOT flipping the whole job status.
        
        # 1. Find or Create Correction Job
        correction_job_id = None
        for jid, j in data["jobs"].items():
            if j.get("parent_id") == str(job_id) and j["status"] == "annotating":
                correction_job_id = jid
                break
                
        if not correction_job_id:
            # Create new job
            correction_job_id = str(data["next_job_id"])
            data["next_job_id"] += 1
            
            new_job = Job(
                id=correction_job_id,
                name=f"{job['name']} (Fixes)",
                status="annotating",
                annotator=job.get("annotator", ""),
                reviewer=job.get("reviewer", ""),
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                stage=job.get("stage", "annotation"),
                parent_id=str(job_id),
                images={}
            )
            data["jobs"][correction_job_id] = new_job.model_dump()
            
            # Create directory
            (self.jobs_dir / correction_job_id / "annotations").mkdir(parents=True, exist_ok=True)

        # 2. Move Image Entry
        # Add to correction job
        c_job = data["jobs"][correction_job_id]
        c_job["images"][image_name] = job["images"][image_name]
        c_job["images"][image_name]["status"] = "rejected"
        if comment:
            c_job["images"][image_name]["comments"].append(comment)
        c_job["updated_at"] = datetime.now().isoformat()
        
        # Remove from current job
        del job["images"][image_name]
        job["updated_at"] = datetime.now().isoformat()
        
        # 3. Update State Registry (re-assign to correction job)
        state = self.load_state()
        if image_name in state["images"]:
             state["images"][image_name]["job_id"] = correction_job_id
             # status remains 'assigned' essentially, but now to a new job
             state["images"][image_name]["last_updated"] = datetime.now().isoformat()
        self.save_state(state)
        
        # No file moving needed!
            
        self.save_jobs(data)



    def get_unassigned_count(self) -> int:
        state = self.load_state()
        if self.sync_images(state):
            self.save_state(state)
            
        count = sum(1 for img in state["images"].values() if img["status"] == "unassigned")
        return count

    def get_dataset_count(self) -> int:
        state = self.load_state()
        count = sum(1 for img in state["images"].values() if img["status"] == "dataset")
        return count

    def reset_dataset(self):
        # Update all 'dataset' images to 'unassigned'
        state = self.load_state()
        count = 0
        for img, info in state["images"].items():
            if info["status"] == "dataset":
                info["status"] = "unassigned"
                info["job_id"] = None
                info["last_updated"] = datetime.now().isoformat()
                count += 1
        
        self.save_state(state)
        # Files remain in annotations/ folder logic. No deletions.
