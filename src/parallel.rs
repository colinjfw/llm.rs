use std::mem::MaybeUninit;

#[cfg(not(feature = "parallel"))]
pub fn parallel<const N: usize, F, R>(actual: usize, func: F) -> [MaybeUninit<R>; N]
where
    F: Fn(usize) -> R + Send + Sync,
    R: Send,
{
    let mut results: [MaybeUninit<R>; N] = [const { MaybeUninit::uninit() }; N];
    for i in 0..actual {
        results[i].write(func(i));
    }
    results
}

#[cfg(feature = "parallel")]
pub fn parallel<const N: usize, F, R>(actual: usize, func: F) -> [MaybeUninit<R>; N]
where
    F: Fn(usize) -> R + Send + Sync,
    R: Send,
{
    use self::parallel::*;

    assert!(actual <= N && N <= THREADS);

    let pool = THREAD_POOL.get_or_init(ThreadPool::spawn);

    let results: [MaybeUninit<R>; N] = [const { MaybeUninit::uninit() }; N];
    let job = Job {
        func,
        results: UnsafeCell::new(results),
        waiter: Waiter::new(N),
    };
    let job_ptr = &job as *const _;

    for i in 0..actual {
        // SAFETY: `i` is only used once and pointer allocated by JobRef lives
        // until `waiter` signals that all jobs are now finished.
        let job_ref = JobRef {
            iter: i,
            pointer: job_ptr as *const (),
            execute_fn: Job::<N, F, R>::execute,
        };
        unsafe {
            pool.threads[i].send(job_ref);
        }
    }

    job.waiter.wait();
    job.results.into_inner()
}

#[cfg(feature = "parallel")]
pub(super) mod parallel {
    use std::{
        cell::UnsafeCell,
        mem::MaybeUninit,
        sync::{Condvar, Mutex, OnceLock},
        thread,
    };

    const THREADS: usize = 32;

    static THREAD_POOL: OnceLock<ThreadPool> = OnceLock::new();

    struct ThreadPool {
        threads: Box<[&'static Thread]>,
    }

    impl ThreadPool {
        fn spawn() -> Self {
            let threads = (0..THREADS)
                .map(|_| {
                    let thread = &*Box::leak(Box::new(Thread {
                        inbox: Mutex::new(None),
                        condvar: Condvar::new(),
                    }));
                    thread::spawn(|| thread.run());
                    thread
                })
                .collect();
            Self { threads }
        }
    }

    struct Thread {
        inbox: Mutex<Option<JobRef>>,
        condvar: Condvar,
    }

    impl Thread {
        /// SAFETY:
        /// * Caller must guarantee that `pointer` is alive and matches `execute_fn`
        /// * Caller must guarantee that `i` is only used once.
        unsafe fn send(&self, job: JobRef) {
            let prev = self.inbox.lock().unwrap().replace(job);
            assert!(prev.is_none(), "job didn't finish");
            self.condvar.notify_all();
        }

        fn run(&self) {
            let mut inbox = self.inbox.lock().expect("thread pool poisoned");
            loop {
                if let Some(job) = inbox.take() {
                    // Safety: Inherits safety properties from `send`.
                    unsafe { (job.execute_fn)(job.pointer, job.iter) }
                }
                inbox = self.condvar.wait(inbox).expect("thread pool poisoned");
            }
        }
    }

    struct Job<const N: usize, F, R> {
        func: F,
        results: UnsafeCell<[MaybeUninit<R>; N]>,
        waiter: Waiter,
    }

    impl<const N: usize, F, R> Job<N, F, R>
    where
        F: Fn(usize) -> R + Send + Sync,
        R: Send,
    {
        /// SAFETY:
        /// * Caller must guarantee that `ptr` is alive and is of type `Self`.
        /// * Caller must guarantee that `i` is only used once.
        unsafe fn execute(ptr: *const (), i: usize) {
            let this = &*(ptr as *const Self);
            let results = &mut *this.results.get();

            let result = (this.func)(i);
            results[i].write(result);

            this.waiter.done();
        }
    }

    struct JobRef {
        iter: usize,
        pointer: *const (),
        execute_fn: unsafe fn(*const (), usize),
    }

    // Safety: parallel() requires pre-erased types to be Send
    unsafe impl Send for JobRef {}

    struct Waiter {
        lock: Mutex<usize>,
        cvar: Condvar,
    }

    impl Waiter {
        fn new(threads: usize) -> Self {
            Self {
                lock: Mutex::new(threads),
                cvar: Condvar::new(),
            }
        }

        fn wait(&self) {
            let mut lock = self.lock.lock().expect("waiter poisoned");
            loop {
                if *lock == 0 {
                    return;
                }
                lock = self.cvar.wait(lock).expect("waiter poisoned");
            }
        }

        fn done(&self) {
            let mut lock = self.lock.lock().unwrap();
            *lock -= 1;
            if *lock == 0 {
                self.cvar.notify_all();
            }
        }
    }
}
