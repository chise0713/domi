use std::{
    cell::RefCell,
    collections::HashSet,
    fmt,
    hash::{BuildHasher, Hash},
    marker::PhantomData,
    panic::{AssertUnwindSafe, catch_unwind},
    rc::Rc,
};

use crate::Hasher;

#[cold]
#[inline(never)]
fn missing_pool() -> ! {
    panic!("intern pool not initialized; missing PoolGuard");
}

#[repr(transparent)]
struct Interner<T>
where
    T: Eq + Hash + ?Sized,
{
    set: HashSet<Rc<T>, Hasher>,
}

impl<T> Interner<T>
where
    T: Eq + Hash + ?Sized,
{
    #[inline]
    fn new() -> Self {
        Self {
            set: HashSet::default(),
        }
    }

    #[inline]
    fn intern(&mut self, s: Rc<T>) -> Rc<T> {
        if let Some(v) = self.set.get(&s) {
            v.clone()
        } else {
            self.set.insert(s.clone());
            s
        }
    }

    #[inline]
    fn intern_ref(&self, value: &T) -> Option<Rc<T>> {
        self.set.get(value).cloned()
    }

    // reduce inc, dec ref count
    #[inline]
    fn intern_id(&self, value: &T) -> Option<InternId> {
        // Safety: it's from interner
        Some(unsafe { InternId::from_interned(self.set.get(value)?) })
    }

    #[inline]
    fn clear(&mut self) {
        self.set.clear();
        self.set.shrink_to_fit();
    }
}

// Discards DST metadata
// saves some stack memory
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub(crate) struct InternId(usize);

impl InternId {
    /// # Safety
    ///
    /// - `value` must originate from the same interner instance
    /// - [`Rc<T>`] identity must not be externally constructed
    /// - pointer is used as stable allocation identity
    #[inline(always)]
    pub(crate) unsafe fn from_interned<T: ?Sized>(value: &Rc<T>) -> Self {
        Self(Rc::as_ptr(value).addr())
    }
}

impl fmt::Debug for InternId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("InternId")
            .field(&(self.0 as *const ()))
            .finish()
    }
}

impl Hash for InternId {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_usize(self.0);
    }
}

#[derive(Default)]
#[repr(transparent)]
pub(crate) struct InternIdHasher {
    state: usize,
}

impl BuildHasher for InternIdHasher {
    type Hasher = InternIdHasher;

    #[inline]
    fn build_hasher(&self) -> Self::Hasher {
        Self::default()
    }
}

impl std::hash::Hasher for InternIdHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.state as u64
    }

    #[inline]
    fn write(&mut self, _: &[u8]) {
        unreachable!("InternIdHasher only supports usize")
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.state = i;
    }
}

macro_rules! define_pools {
    (
        $(
            $name:ident : $ty:ty
        ),* $(,)?
    ) => {
        ::paste::paste! {
            struct PoolState {
                used_count: usize,
                $(
                    [<$name:snake>]: Interner<$ty>,
                )*
            }

            impl PoolState {
                fn new() -> Self {
                    Self {
                        used_count: 0,
                        $(
                            [<$name:snake>]: Interner::new(),
                        )*
                    }
                }

                fn acquire(&mut self) {
                    assert!(self.used_count != usize::MAX, "Pool is poisoned due to previous panic in Drop");

                    self.used_count += 1;
                }

                fn release(&mut self) {
                    if self.used_count == usize::MAX {
                        return;
                    }

                    assert!(self.used_count > 0, "PoolState.used_count underflow");

                    self.used_count -= 1;
                    if self.used_count > 0 {
                        return;
                    }

                    if catch_unwind(AssertUnwindSafe(|| self.clear())).is_err() {
                        self.used_count = usize::MAX;
                    } else {
                        self.used_count = 0;
                    };
                }

                fn clear(&mut self) {
                    #[cfg(test)]
                    if FORCE_CLEAR_PANIC.get() {
                        panic!("forced");
                    };

                    $(
                        self.[<$name:snake>].clear();
                    )*
                }
            }

            thread_local! {
                static POOL_STATE: RefCell<PoolState> = RefCell::new(PoolState::new());
            }

            $(
                pub(crate) struct [<$name Pool>];

                impl [<$name Pool>] {
                    #[inline]
                    pub(crate) fn [<$name:snake>](value: Rc<$ty>) -> Rc<$ty> {
                        POOL_STATE.with_borrow_mut(|p| {
                            if p.used_count == 0 {
                                missing_pool()
                            }
                            p.[<$name:snake>].intern(value)
                        })
                    }

                    #[inline]
                    pub(crate) fn [<$name:snake _ref>](value: &$ty) -> Option<Rc<$ty>> {
                        POOL_STATE.with_borrow(|p| {
                            if p.used_count == 0 {
                                missing_pool()
                            }
                            p.[<$name:snake>].intern_ref(value)
                        })
                    }
                }
            )*
        }
    };
}

define_pools! {
    Base: str,
    Attr: str,
    DomainValue: str,
    AttrSlice: [Rc<str>],
}

impl BasePool {
    #[inline]
    pub(crate) fn base_id(value: &str) -> Option<InternId> {
        POOL_STATE.with_borrow(|p| p.base.intern_id(value))
    }
}

#[cfg(test)]
thread_local! {
    static FORCE_CLEAR_PANIC: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

macro_rules! maybe_intern {
    ($use_pool:expr, $s:expr, $name:ident) => {
        if !$use_pool {
            ::std::rc::Rc::from($s)
        } else {
            $crate::interner::intern!($s, $name)
        }
    };
}

macro_rules! intern {
    ($s:expr, $name:ident) => {
        ::paste::paste! {
            $crate::interner::[<$name Pool>]::[<$name:snake _ref>]($s.as_ref())
                .unwrap_or_else(|| {
                    $crate::interner::[<$name Pool>]::[<$name:snake>]
                        (::std::rc::Rc::from($s))
                })
        }
    };
}

pub(crate) use intern;
pub(crate) use maybe_intern;

type NotSyncNorSend = PhantomData<Rc<()>>;

pub(crate) struct PoolGuard {
    _marker: NotSyncNorSend,
}

impl PoolGuard {
    pub(crate) fn acquire() -> Self {
        POOL_STATE.with_borrow_mut(PoolState::acquire);
        Self {
            _marker: NotSyncNorSend::default(),
        }
    }
}

impl Default for PoolGuard {
    fn default() -> Self {
        Self::acquire()
    }
}

impl Drop for PoolGuard {
    fn drop(&mut self) {
        POOL_STATE.with_borrow_mut(PoolState::release)
    }
}

impl fmt::Debug for PoolGuard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PoolGuard").finish()
    }
}

#[derive(Clone, Copy)]
#[repr(transparent)]
pub(crate) struct PackedAttr {
    inner: usize,
}

impl PackedAttr {
    const TAG_MASK: usize = 0x1;
    const ADDR_MASK: usize = !Self::TAG_MASK;

    #[inline(always)]
    pub(crate) const fn new(attr_id: InternId, tag: bool) -> Self {
        let InternId(ptr_addr) = attr_id;
        assert!(ptr_addr.is_multiple_of(2));
        Self {
            inner: ptr_addr | tag as usize,
        }
    }

    #[inline(always)]
    pub(crate) const fn addr(&self) -> usize {
        self.inner & Self::ADDR_MASK
    }

    #[inline(always)]
    pub(crate) const fn tag(&self) -> bool {
        (self.inner & Self::TAG_MASK) != 0
    }
}

impl fmt::Debug for PackedAttr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PackedAttr")
            .field("addr", &(self.addr() as *const ()))
            .field("tag", &self.tag())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_attr_logic() {
        let original_addr = 0x12345670_usize;
        let attr_id = InternId(original_addr);

        let packed_true = PackedAttr::new(attr_id, true);
        assert!(packed_true.tag());
        assert_eq!(packed_true.addr(), original_addr);
        assert_eq!(packed_true.inner, original_addr | 1);

        let packed_false = PackedAttr::new(attr_id, false);
        assert!(!packed_false.tag());
        assert_eq!(packed_false.addr(), original_addr);
        assert_eq!(packed_false.inner, original_addr);
    }

    #[test]
    fn with_real_pointer() {
        let val = Box::new(42);
        let ptr_addr = &*val as *const i32 as usize;

        assert_eq!(ptr_addr & 0x1, 0);

        let attr_id = InternId(ptr_addr);

        let p1 = PackedAttr::new(attr_id, true);
        let p2 = PackedAttr::new(attr_id, false);

        assert!(p1.tag());
        assert!(!p2.tag());
        assert_eq!(p1.addr(), ptr_addr);
        assert_eq!(p2.addr(), ptr_addr);
    }

    #[test]
    fn const_capability() {
        const ADDR: InternId = InternId(0x1000);
        const PACKED: PackedAttr = PackedAttr::new(ADDR, true);

        const _: () = assert!(PACKED.tag());
        assert_eq!(PACKED.addr(), 0x1000);
    }

    #[test]
    fn intern_id_hasher() {
        use std::assert_matches;

        let mut h: HashSet<usize, InternIdHasher> = HashSet::default();
        h.insert(1);
        h.insert(2);

        let mut h: HashSet<u8, InternIdHasher> = HashSet::default();
        assert_matches!(catch_unwind(AssertUnwindSafe(|| h.insert(1))), Err(_));
    }

    #[test]
    fn pool_clear_panic() {
        // no pool guard
        assert!(catch_unwind(|| BasePool::base(Rc::from(""))).is_err());

        let pg = PoolGuard::acquire();
        // tests are running from different threads,
        // even if `--test-threads=1` is passed, so no
        // worries about poison the testing environment
        FORCE_CLEAR_PANIC.set(true);
        drop(pg); // <-- trigger a `Drop` panic

        assert!(catch_unwind(PoolGuard::acquire).is_err());
        assert!(catch_unwind(PoolGuard::acquire).is_err());
    }

    #[test]
    fn pool_clear() {
        {
            let _pg = PoolGuard::acquire();

            BasePool::base(Rc::from("abc"));
            assert!(BasePool::base_ref("abc").is_some());
        }

        {
            let _pg = PoolGuard::acquire();

            assert!(BasePool::base_ref("abc").is_none());
        }
    }

    #[test]
    fn nested_pool_guard() {
        let pg1 = PoolGuard::acquire();
        let pg2 = PoolGuard::acquire();

        BasePool::base(Rc::from("hello"));

        drop(pg2);

        BasePool::base(Rc::from("hello"));

        drop(pg1);

        assert!(catch_unwind(|| BasePool::base(Rc::from("hello"))).is_err());
    }

    #[test]
    fn intern() {
        let _g = PoolGuard::acquire();

        let a = BasePool::base(Rc::from("abc"));
        let b = BasePool::base(Rc::from("abc"));

        assert!(Rc::ptr_eq(&a, &b));

        assert_eq!(Rc::strong_count(&a), 3);
    }

    #[test]
    fn intern_ref() {
        let _pg = PoolGuard::acquire();

        assert!(BasePool::base_ref("abc").is_none());

        let a = BasePool::base(Rc::from("abc"));

        let b = BasePool::base_ref("abc").unwrap();

        assert!(Rc::ptr_eq(&a, &b));
    }

    #[test]
    fn intern_id() {
        let _pg = PoolGuard::acquire();

        BasePool::base(Rc::from("abc"));

        let a = BasePool::base_id("abc").unwrap();
        let b = BasePool::base_id("abc").unwrap();

        assert_eq!(a, b);

        assert!(BasePool::base_id("def").is_none());
    }

    #[test]
    fn pool_underflow() {
        assert!(
            catch_unwind(|| {
                let mut s = PoolState::new();
                s.release()
            })
            .is_err()
        );
    }
}
