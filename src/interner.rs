use std::{
    cell::{Cell, RefCell},
    collections::HashSet,
    hash::{BuildHasher, Hash},
    marker::PhantomData,
    panic::catch_unwind,
    rc::Rc,
};

use crate::Hasher;

struct Interner<T>
where
    T: Eq + Hash + ?Sized,
{
    set: Option<HashSet<Rc<T>, Hasher>>,
}

impl<T> Interner<T>
where
    T: Eq + Hash + ?Sized,
{
    const fn new() -> Self {
        Self { set: None }
    }

    fn initialize(&mut self) {
        self.set = Some(HashSet::default())
    }

    fn intern(&mut self, s: Rc<T>) -> Rc<T> {
        let set = self
            .set
            .as_mut()
            .expect("intern pool not initialized; missing PoolGuard");
        if let Some(v) = set.get(&s) {
            v.clone()
        } else {
            set.insert(s.clone());
            s
        }
    }

    fn intern_ref(&self, value: &T) -> Option<Rc<T>> {
        let set = self.set.as_ref()?;
        set.get(value).cloned()
    }

    // reduce inc, dec ref count
    fn intern_id(&self, value: &T) -> Option<InternId> {
        let set = self.set.as_ref()?;

        // Safety: it's from interner
        Some(unsafe { InternId::from_interned(set.get(value)?) })
    }

    fn clear(&mut self) {
        self.set = None;
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

impl std::fmt::Debug for InternId {
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

macro_rules! define_pool {
    ($name:ident, $ty:ty) => {
        ::paste::paste! {
            thread_local! {
                static [< $name:snake:upper _POOL >]: RefCell<Interner<$ty>> = const { RefCell::new(Interner::new()) };
            }
            pub(crate) struct [< $name Pool >];
            impl [< $name Pool >] {
                fn initialize() {
                    [< $name:snake:upper _POOL >].with(|p| p.borrow_mut().initialize())
                }
                pub(crate) fn [< $name:snake >](value: Rc<$ty>) -> Rc<$ty> {
                    [< $name:snake:upper _POOL >].with(|p| p.borrow_mut().intern(value))
                }
                pub(crate) fn [< $name:snake _ref >](value: &$ty) -> Option<Rc<$ty>> {
                    [< $name:snake:upper _POOL >].with(|p| p.borrow().intern_ref(value))
                }
                fn clear() {
                    [< $name:snake:upper _POOL >].with(|p| p.borrow_mut().clear())
                }
            }
        }
    };
}

define_pool!(Base, str);
define_pool!(Attr, str);
define_pool!(DomainValue, str);
define_pool!(AttrSlice, [Rc<str>]);

impl BasePool {
    pub(crate) fn base_id(value: &str) -> Option<InternId> {
        BASE_POOL.with(|p| p.borrow().intern_id(value))
    }
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

thread_local! {
    static POOL_USED_COUNT: Cell<isize> = const { Cell::new(0) };
}

type NotSyncNorSend = PhantomData<Rc<()>>;

#[derive(Debug)]
pub(crate) struct PoolGuard {
    _marker: NotSyncNorSend,
}

impl PoolGuard {
    pub(crate) fn acquire() -> Self {
        let n = POOL_USED_COUNT.get();
        if n <= 0 {
            POOL_USED_COUNT.set(1);
            #[cfg(debug_assertions)]
            if n < 0 {
                dbg!("POOL_USED_COUNT underflow", n);
            }
            Self::clear_pools();
            AttrPool::initialize();
            BasePool::initialize();
            DomainValuePool::initialize();
            AttrSlicePool::initialize();
        } else if n == isize::MAX {
            panic!("Pool is poisoned due to previous panic in Drop");
        } else {
            POOL_USED_COUNT.set(n + 1);
        }
        Self {
            _marker: NotSyncNorSend::default(),
        }
    }

    fn clear_pools() {
        AttrPool::clear();
        BasePool::clear();
        DomainValuePool::clear();
        AttrSlicePool::clear();
    }
}

impl Default for PoolGuard {
    fn default() -> Self {
        Self::acquire()
    }
}

impl Drop for PoolGuard {
    fn drop(&mut self) {
        let n = POOL_USED_COUNT.get() - 1;
        if n <= 0 {
            POOL_USED_COUNT.set(0);
            if catch_unwind(Self::clear_pools).is_err() {
                POOL_USED_COUNT.set(isize::MAX);
            };
            #[cfg(debug_assertions)]
            if n < 0 {
                dbg!("POOL_USED_COUNT underflow", n);
            }
        } else {
            POOL_USED_COUNT.set(n);
        }
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

impl std::fmt::Debug for PackedAttr {
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
        use std::{
            assert_matches,
            panic::{AssertUnwindSafe, catch_unwind},
        };

        let mut h: HashSet<usize, InternIdHasher> = HashSet::default();
        h.insert(1);
        h.insert(2);

        let mut h: HashSet<u8, InternIdHasher> = HashSet::default();
        assert_matches!(catch_unwind(AssertUnwindSafe(|| h.insert(1))), Err(_));
    }
}
